#!/usr/bin/env python3
"""
Behavior Cloning (BC) training that is CHECKPOINT-COMPATIBLE with your collector.

Key compatibility changes vs your current bc.py:
1) Train an Actor that matches the collector's Actor architecture:
   - fc1, fc2, fc_mean, fc_logstd
   - buffers: action_scale, action_bias
   - checkpoint key "actor" is now loadable by build_actor_from_ckpt(...)

2) Include collector-relevant fields in ckpt["config"]:
   episode_length, discrete_action, y_clip, logdet_eps, max_episode_steps

Run example:
  python3 online_RL/bc.py \
    --dataset_path hand_dataset.npy \
    --env_name GuidanceCorridorMaze \
    --start "2.5,14.5" \
    --goal "14.5,2.5" \
    --reward_type sparse \
    --terminate_on_wall False \
    --discretize_eval True \
    --episode_length 200 \
    --max_episode_steps 512 \
    --epochs 200 \
    --eval_every_epochs 10 \
    --save_model True \
    --save_every_epochs 50 \
    --checkpoints_path discrete_GuidanceCorridorMaze/bc_seed_100
"""

import os, sys
import math
import random
import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyrallis
import imageio
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PointMassEnv import PointMassEnv


# -----------------------------
# Utils
# -----------------------------
def parse_vec2(s: str) -> np.ndarray:
    s = s.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    x, y = [float(v) for v in s.split(",")]
    return np.array([x, y], dtype=np.float32)


def set_seed(seed: int, torch_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def load_dataset(dataset_path: str) -> dict:
    # First try pickle (your collector writes pickle)
    try:
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "observations" in data and "actions" in data:
            return data
    except Exception as e:
        print(f"[load_dataset] pickle load failed: {e}")

    # Fallback: numpy object that contains a dict
    try:
        obj = np.load(dataset_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
            data = obj.item()
            if "observations" in data and "actions" in data:
                return data
    except Exception as e:
        print(f"[load_dataset] np.load fallback failed: {e}")

    raise RuntimeError(f"Could not load dataset from {dataset_path} (expected pickled dict or npy-dict).")


def discretize_action_np(action: np.ndarray) -> np.ndarray:
    """
    Convert continuous action to {-1,0,1} with thresholds +/-0.5.
    action: (2,) or (N,2)
    """
    a = np.array(action, dtype=np.float32, copy=True)
    if a.ndim == 1:
        a = a[None, :]
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        x, y = float(a[i, 0]), float(a[i, 1])
        if x < -0.5:
            xq = -1.0
        elif x > 0.5:
            xq = 1.0
        else:
            xq = 0.0
        if y < -0.5:
            yq = -1.0
        elif y > 0.5:
            yq = 1.0
        else:
            yq = 0.0
        out[i] = [xq, yq]
    return out[0] if action.ndim == 1 else out


class BCDataset(torch.utils.data.Dataset):
    def __init__(self, obs: np.ndarray, acts: np.ndarray):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


# -----------------------------
# Actor = EXACTLY compatible with your collector
# -----------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -5


def to_np(x):
    return np.asarray(x, dtype=np.float32)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_mean = nn.Linear(self.hidden_dim, self.act_dim)
        self.fc_logstd = nn.Linear(self.hidden_dim, self.act_dim)

        act_low = to_np(act_low).reshape(-1)
        act_high = to_np(act_high).reshape(-1)
        if act_low.shape[0] != self.act_dim or act_high.shape[0] != self.act_dim:
            raise ValueError(f"act_low/high must be shape ({self.act_dim},), got {act_low.shape} {act_high.shape}")

        action_scale = (act_high - act_low) / 2.0
        action_bias = (act_high + act_low) / 2.0
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    # Differentiable deterministic action (for BC loss)
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        y = torch.tanh(mean)
        return y * self.action_scale + self.action_bias

    # Inference helper (no_grad)
    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.deterministic_action(obs)


# -----------------------------
# Args
# -----------------------------
@dataclass
class Args:
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = True
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None

    checkpoints_path: str = "checkpoints/bc_run"
    save_model: bool = True
    save_every_epochs: int = 50

    # dataset
    dataset_path: str = "hand_dataset.npy"
    val_ratio: float = 0.1

    # env / eval rollout
    env_name: str = "GuidanceCorridorMaze"
    reward_type: str = "sparse"
    terminate_on_wall: bool = False

    # IMPORTANT: this controls whether rollout uses discretized {-1,0,1} moves.
    discretize_eval: bool = True

    # Add these so collector can reuse them from ckpt["config"]
    episode_length: int = 200
    max_episode_steps: int = 512   # rollout safety cap (also used by collector)
    y_clip: float = 1.0 - 1e-5
    logdet_eps: float = 1e-5

    # Collector expects "discrete_action" in config; we mirror discretize_eval into it at runtime.
    discrete_action: bool = True

    start: str = "2.5,14.5"
    goal: str = "14.5,2.5"
    goal_radius: float = 0.8

    # BC training
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden: int = 256

    # evaluation cadence (gif)
    eval_every_epochs: int = 10
    eval_episodes: int = 1


@torch.no_grad()
def eval_on_dataset(policy: Actor, loader, device, discretize: bool):
    policy.eval()
    total_mse = 0.0
    total_n = 0
    correct = 0
    total = 0

    for obs, act in loader:
        obs = obs.to(device)
        act = act.to(device)

        pred = policy.deterministic(obs)

        total_mse += F.mse_loss(pred, act, reduction="sum").item()
        total_n += obs.shape[0]

        if discretize:
            pred_q = discretize_action_np(pred.detach().cpu().numpy())
            act_q = discretize_action_np(act.detach().cpu().numpy())
            matches = np.all(pred_q == act_q, axis=1)
            correct += int(matches.sum())
            total += int(matches.shape[0])

    avg_mse = total_mse / max(1, total_n)
    acc = (correct / max(1, total)) if discretize else None
    policy.train()
    return avg_mse, acc


@torch.no_grad()
def rollout_and_save_gif(
    policy: Actor,
    device,
    args: Args,
    start_np: np.ndarray,
    goal_np: np.ndarray,
    gif_path: str,
):
    env = PointMassEnv(
        start=start_np,
        goal=goal_np,
        goal_radius=args.goal_radius,
        env_name=args.env_name,
        terminate_on_wall=args.terminate_on_wall,
        reward_type=args.reward_type,
        episode_length=int(args.episode_length),
    )

    policy.eval()
    images = []
    count_success = 0

    for _ in range(args.eval_episodes):
        obs, _ = env.reset()
        images.append(np.moveaxis(np.transpose(env.render()), 0, -1))

        ep_len = 0
        done = False

        while not done and ep_len < int(args.max_episode_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act = policy.deterministic(obs_t).squeeze(0).detach().cpu().numpy()

            if args.discretize_eval:
                act = discretize_action_np(act)

            obs, reward, terminated, truncated, info = env.step(act)
            images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
            ep_len += 1
            done = bool(terminated or truncated)

            if terminated and isinstance(info, dict) and info.get("success", False):
                count_success += 1

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, images, fps=10)

    policy.train()
    success_rate = count_success / float(args.eval_episodes)
    return {"success_rate": success_rate}


def main():
    args = pyrallis.parse(config_class=Args)

    # Mirror rollout discretization into the config field the collector expects.
    args.discrete_action = bool(args.discretize_eval)

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    # --- dirs ---
    run_name = args.checkpoints_path
    writer = SummaryWriter(f"bc/{run_name}")
    gif_dir = f"bc/{run_name}/gifs"
    model_dir = f"bc/{run_name}/models"
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # wandb (optional)
    wandb = None
    if args.track:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # --- load dataset ---
    data = load_dataset(args.dataset_path)
    obs = np.asarray(data["observations"], dtype=np.float32)
    acts = np.asarray(data["actions"], dtype=np.float32)

    if obs.ndim != 2:
        raise ValueError(f"Expected observations shape (N,obs_dim), got {obs.shape}")
    if acts.ndim != 2:
        raise ValueError(f"Expected actions shape (N,act_dim), got {acts.shape}")
    if obs.shape[0] != acts.shape[0]:
        raise ValueError(f"obs/actions length mismatch: {obs.shape[0]} vs {acts.shape[0]}")

    # quick sanity print
    print(f"[data] obs: shape={obs.shape}, min={obs.min(axis=0)}, max={obs.max(axis=0)}")
    print(f"[data] act: shape={acts.shape}, min={acts.min(axis=0)}, max={acts.max(axis=0)}")

    N = obs.shape[0]
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(N)
    n_val = int(math.ceil(args.val_ratio * N))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    train_ds = BCDataset(obs[tr_idx], acts[tr_idx])
    val_ds = BCDataset(obs[val_idx], acts[val_idx])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print(f"[device] {device}")
    print(f"[data] N={N} train={len(train_ds)} val={len(val_ds)}")

    start_np = parse_vec2(args.start)
    goal_np = parse_vec2(args.goal)

    # derive policy_spec from env instance (dims + bounds)
    tmp_env = PointMassEnv(
        start=start_np,
        goal=goal_np,
        goal_radius=args.goal_radius,
        env_name=args.env_name,
        terminate_on_wall=args.terminate_on_wall,
        reward_type=args.reward_type,
        episode_length=int(args.episode_length),
    )
    obs_dim = int(np.array(tmp_env.observation_space.shape).prod())
    act_dim = int(np.prod(tmp_env.action_space.shape))
    act_low = np.asarray(tmp_env.action_space.low, dtype=np.float32).reshape(-1)
    act_high = np.asarray(tmp_env.action_space.high, dtype=np.float32).reshape(-1)
    tmp_env.close()

    # Ensure dataset dims match env dims (prevents silent mismatch)
    if obs.shape[1] != obs_dim:
        raise ValueError(f"Dataset obs_dim={obs.shape[1]} but env obs_dim={obs_dim}")
    if acts.shape[1] != act_dim:
        raise ValueError(f"Dataset act_dim={acts.shape[1]} but env act_dim={act_dim}")

    policy_spec = {
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "hidden_dim": int(args.hidden),
        "act_low": act_low,
        "act_high": act_high,
    }

    # --- model / opt (Actor compatible with collector) ---
    policy = Actor(
        obs_dim=policy_spec["obs_dim"],
        act_dim=policy_spec["act_dim"],
        act_low=policy_spec["act_low"],
        act_high=policy_spec["act_high"],
        hidden_dim=args.hidden,
    ).to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- training loop ---
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = 0.0
        seen = 0

        for ob_b, ac_b in train_loader:
            ob_b = ob_b.to(device)
            ac_b = ac_b.to(device)

            # BC loss on deterministic action (DIFFERENTIABLE)
            pred = policy.deterministic_action(ob_b)
            loss = F.mse_loss(pred, ac_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * ob_b.shape[0]
            seen += ob_b.shape[0]
            global_step += 1

        train_mse = running / max(1, seen)
        val_mse, val_acc = eval_on_dataset(policy, val_loader, device, discretize=args.discretize_eval)

        writer.add_scalar("losses/train_mse", train_mse, epoch)
        writer.add_scalar("losses/val_mse", val_mse, epoch)
        if val_acc is not None:
            writer.add_scalar("metrics/val_discrete_match", val_acc, epoch)

        if wandb is not None:
            log_dict = {"train_mse": train_mse, "val_mse": val_mse, "epoch": epoch}
            if val_acc is not None:
                log_dict["val_discrete_match"] = val_acc
            wandb.log(log_dict, step=epoch)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            msg = f"[epoch {epoch:03d}] train_mse={train_mse:.6f} val_mse={val_mse:.6f}"
            if val_acc is not None:
                msg += f" val_discrete_match={val_acc*100:.2f}%"
            print(msg)

        # --- periodic rollout GIF eval ---
        if args.eval_every_epochs > 0 and (epoch % args.eval_every_epochs == 0):
            gif_path = os.path.join(gif_dir, f"{epoch}.gif")
            stats = rollout_and_save_gif(policy, device, args, start_np, goal_np, gif_path)

            writer.add_scalar("eval/success_rate", stats["success_rate"], epoch)
            if wandb is not None:
                wandb.log(
                    {
                        "eval_success_rate": stats["success_rate"],
                        "policy_rollout": wandb.Video(gif_path, fps=10, format="gif"),
                        "epoch": epoch,
                    },
                    step=epoch,
                )
            print(f"[eval] epoch={epoch} success_rate={stats['success_rate']:.3f} gif={gif_path}")

        # --- periodic checkpoint (collector-compatible) ---
        if args.save_model and args.save_every_epochs > 0 and (epoch % args.save_every_epochs == 0):
            ckpt_path = os.path.join(model_dir, f"{epoch}_bc_policy.pth")
            payload = {
                "version": "bc_pointmass_ckpt_v2_actor_compatible",
                "time": time.time(),
                "config": vars(args),  # includes episode_length, discrete_action, y_clip, logdet_eps, max_episode_steps
                "policy_spec": {
                    "obs_dim": int(policy_spec["obs_dim"]),
                    "act_dim": int(policy_spec["act_dim"]),
                    "hidden_dim": int(policy_spec.get("hidden_dim", args.hidden)),
                    "act_low": np.asarray(policy_spec["act_low"], dtype=np.float32),
                    "act_high": np.asarray(policy_spec["act_high"], dtype=np.float32),
                },
                "actor": policy.state_dict(),          # <-- matches collector Actor
                "optimizer": opt.state_dict(),
                "global_step": int(global_step),
                "epoch": int(epoch),
            }
            torch.save(payload, ckpt_path)
            print(f"[saved] {ckpt_path}")

    # final save
    if args.save_model:
        final_path = os.path.join(model_dir, "final_bc_policy.pth")
        payload = {
            "version": "bc_pointmass_ckpt_v2_actor_compatible",
            "time": time.time(),
            "config": vars(args),
            "policy_spec": {
                "obs_dim": int(policy_spec["obs_dim"]),
                "act_dim": int(policy_spec["act_dim"]),
                "hidden_dim": int(policy_spec.get("hidden_dim", args.hidden)),
                "act_low": np.asarray(policy_spec["act_low"], dtype=np.float32),
                "act_high": np.asarray(policy_spec["act_high"], dtype=np.float32),
            },
            "actor": policy.state_dict(),
            "optimizer": opt.state_dict(),
            "global_step": int(global_step),
            "epoch": int(args.epochs),
        }
        torch.save(payload, final_path)
        print(f"[saved] {final_path}")

    writer.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
