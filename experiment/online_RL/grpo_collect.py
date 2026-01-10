import os
import sys
import time
import random
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Make sure PointMassEnv is importable (match your project layout)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from PointMassEnv import PointMassEnv  # noqa: E402


# -----------------------------
# Utilities (mirrors grpo.py)
# -----------------------------
def to_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def parse_vec2(s: str) -> np.ndarray:
    s = s.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    x, y = [float(v) for v in s.split(",")]
    return np.array([x, y], dtype=np.float32)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Discrete action mapping (same as grpo.py)
# -----------------------------
def discrete_action_fn(action: np.ndarray) -> np.ndarray:
    if action.ndim == 1:
        action = action[None, :]
    out = action.copy()
    for i in range(out.shape[0]):
        x, y = out[i]
        x = -1 if x < -0.5 else (1 if x > 0.5 else 0)
        y = -1 if y < -0.5 else (1 if y > 0.5 else 0)
        out[i] = [x, y]
    return out


# -----------------------------
# Actor (copied from grpo.py, env-free)
# -----------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -5


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


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

    @staticmethod
    def from_spec(spec: Dict[str, Any]) -> "Actor":
        return Actor(
            obs_dim=int(spec["obs_dim"]),
            act_dim=int(spec["act_dim"]),
            act_low=to_np(spec["act_low"]),
            act_high=to_np(spec["act_high"]),
            hidden_dim=int(spec.get("hidden_dim", 256)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std, validate_args=False)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, y_clip: float, logdet_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        y_t = torch.clamp(y_t, -y_clip, y_clip)
        action = y_t * self.action_scale + self.action_bias

        logp = dist.log_prob(x_t)
        log_det = torch.log(self.action_scale) + torch.log(torch.clamp(1 - y_t.pow(2), min=logdet_eps))
        logp = (logp - log_det).sum(-1)
        return action, logp

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        y = torch.tanh(mean)
        return y * self.action_scale + self.action_bias


# -----------------------------
# Checkpoint loading (compatible with your grpo.py)
# -----------------------------
def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    # PyTorch 2.6+ defaults weights_only=True, which breaks ckpts containing numpy objects.
    # If you trust the checkpoint source (you created it), set weights_only=False.
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only kwarg
        ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")
    return ckpt


def build_actor_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> Tuple[Actor, Dict[str, Any]]:
    spec = ckpt.get("policy_spec", None)
    if spec is None:
        raise KeyError("Checkpoint missing 'policy_spec' (cannot reconstruct Actor).")

    spec = dict(spec)
    spec["act_low"] = to_np(spec["act_low"])
    spec["act_high"] = to_np(spec["act_high"])

    actor = Actor.from_spec(spec).to(device)
    actor.load_state_dict(ckpt["actor"], strict=True)
    actor.eval()
    return actor, spec


# -----------------------------
# Args for collection
# -----------------------------
@dataclass
class CollectArgs:
    checkpoint_path: str = "path/to/checkpoint.pt"
    out_path: str = "dataset_collected.pkl"

    # If True, prefer env params from checkpoint config (if present)
    use_ckpt_env: bool = True

    # Env params (used if use_ckpt_env=False OR missing in ckpt config)
    env_name: str = "EmptyRoom"
    reward_type: str = "sparse"
    episode_length: int = 120
    max_episode_steps: int = 512
    start: str = "12.5,4.5"
    goal: str = "4.5,12.5"
    goal_radius: float = 0.8

    # Action behavior
    discrete_action: bool = True
    deterministic_base: bool = True  # base action from actor.deterministic (else actor.sample)
    # For discrete: with probability p, override action with random [-1,0,1]^2
    discrete_random_prob: float = 0.83
    # For continuous: add Gaussian noise ~ N(0, (noise_scale / epi_len)^2) elementwise
    continuous_noise_scale: float = 20.0

    # Collection budget
    seed: int = 0
    traj_num: int = 10_000_000  # big number; we stop by transition_num anyway
    transition_num: int = 5000  # stop after this many transitions

    # Numeric stability (should match training config; we read from ckpt config when available)
    y_clip: float = 1.0 - 1e-5
    logdet_eps: float = 1e-5

    # Device
    cuda: bool = True


def merge_env_from_ckpt_cfg(args: CollectArgs, ckpt_cfg: Dict[str, Any]) -> None:
    # Best-effort: only overwrite env-related fields if present
    for k in ["env_name", "reward_type", "episode_length", "start", "goal", "goal_radius", "discrete_action",
              "y_clip", "logdet_eps", "max_episode_steps"]:
        if k in ckpt_cfg and hasattr(args, k):
            try:
                setattr(args, k, ckpt_cfg[k])
            except Exception:
                pass


# -----------------------------
# Rollout + dataset collection
# -----------------------------
def make_env_from_args(args: CollectArgs) -> PointMassEnv:
    start_np = parse_vec2(args.start)
    goal_np = parse_vec2(args.goal)
    env = PointMassEnv(
        start=start_np,
        goal=goal_np,
        goal_radius=float(args.goal_radius),
        env_name=str(args.env_name),
        reward_type=str(args.reward_type),
        episode_length=int(args.episode_length),
    )
    return env


def pick_action(
    actor: Actor,
    obs: np.ndarray,
    device: torch.device,
    args: CollectArgs,
    epi_len: int,
) -> np.ndarray:
    obs_t = torch.tensor(to_np(obs), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        if args.deterministic_base:
            a = actor.deterministic(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            a, _ = actor.sample(obs_t, args.y_clip, args.logdet_eps)
            a = a.squeeze(0).cpu().numpy().astype(np.float32)

    if args.discrete_action:
        a = discrete_action_fn(a)[0].astype(np.float32)
        # match your previous “inject random discrete move” behavior
        if np.random.rand() < float(args.discrete_random_prob):
            a[0] = np.random.choice([-1.0, 0.0, 1.0])
            a[1] = np.random.choice([-1.0, 0.0, 1.0])
        # a[0] = np.random.choice([-1.0, 0.0, 1.0])
        # a[1] = np.random.choice([-1.0, 0.0, 1.0])
        return a.astype(np.float32)

    # continuous noise like your old script (scale / epi_len)
    noise_std = float(args.continuous_noise_scale) / float(max(1, epi_len))
    a = np.random.normal(loc=a, scale=noise_std).astype(np.float32)
    return a


def collect_dataset(actor: Actor, args: CollectArgs, device: torch.device) -> Dict[str, np.ndarray]:
    env = make_env_from_args(args)

    dataset: Dict[str, List[Any]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminals": [],
        "success": [],
    }

    actor.eval()
    count_success = 0
    transition_count = 0

    for traj_i in range(int(args.traj_num)):
        obs_array = []
        action_array = []
        reward_array = []
        next_obs_array = []
        terminal_array = []
        success_array = []

        obs, _ = env.reset(seed=int(args.seed + traj_i))
        done = False
        epi_len = 1

        while not done and epi_len <= int(args.max_episode_steps):
            action = pick_action(actor, obs, device, args, epi_len)

            obs_array.append(obs.astype(np.float32))
            action_array.append(action.astype(np.float32))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)

            reward_array.append(float(reward))
            next_obs_array.append(next_obs.astype(np.float32))
            terminal_array.append(1.0 if done else 0.0)

            transition_count += 1
            epi_len += 1

            if done and isinstance(info, dict) and bool(info.get("success", False)):
                count_success += 1
                print(f"[traj {traj_i}] SUCCESS length={epi_len}")
                success_array.append(True)

                # preserve your old special-case behavior
                if epi_len != 8:
                    dataset["observations"] += obs_array
                    dataset["actions"] += action_array
                    dataset["rewards"] += reward_array
                    dataset["next_observations"] += next_obs_array
                    dataset["terminals"] += terminal_array

                    success_array = [True] * len(success_array)
                    dataset["success"] += success_array
                else:
                    print("[traj] rarely good !!!! (epi_len==8) -> skipping append like old script")
            elif done:
                dataset["observations"] += obs_array
                dataset["actions"] += action_array
                dataset["rewards"] += reward_array
                dataset["next_observations"] += next_obs_array
                dataset["terminals"] += terminal_array
                dataset["success"] += success_array
            else:
                success_array.append(False)

            obs = next_obs

            if transition_count >= int(args.transition_num):
                break

        print(f"success count: {count_success}")
        print(f"num of trans:  {transition_count}\n")

        if transition_count >= int(args.transition_num):
            break

    env.close()

    out = {
        "observations": np.asarray(dataset["observations"], dtype=np.float32),
        "actions": np.asarray(dataset["actions"], dtype=np.float32),
        "rewards": np.asarray(dataset["rewards"], dtype=np.float32),
        "next_observations": np.asarray(dataset["next_observations"], dtype=np.float32),
        "terminals": np.asarray(dataset["terminals"], dtype=np.float32),
        "success": np.asarray(dataset["success"], dtype=np.float32),
    }
    out["observations"] = np.round(out["observations"], 1)
    out["next_observations"] = np.round(out["next_observations"], 1)
    return out


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Minimal argparse (full script; no pyrallis dependency required here)
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, default="/media/shuozhe/Disk_Bottom_4TB/pointmas_grpo/bc/mislead_BC/mislead_BC_env_GuidanceCorridorMaze_reward_type_sparse_seed_100_GPU_0_bs_64_lr_3e-4_ep_20000_st_2_5__2_5_gl_14_5__2_5/models/1000_bc_policy.pth")
    p.add_argument("--out_path", type=str, default="bc_dataset_collected.pkl")

    p.add_argument("--use_ckpt_env", type=int, default=1)

    p.add_argument("--env_name", type=str, default="GuidanceCorridorMaze")
    p.add_argument("--reward_type", type=str, default="sparse")
    p.add_argument("--episode_length", type=int, default=120)
    p.add_argument("--max_episode_steps", type=int, default=512)
    p.add_argument("--start", type=str, default="2.5,2.5")
    p.add_argument("--goal", type=str, default="14.5,2.5")
    p.add_argument("--goal_radius", type=float, default=0.8)

    p.add_argument("--discrete_action", type=int, default=1)
    p.add_argument("--deterministic_base", type=int, default=1)
    p.add_argument("--discrete_random_prob", type=float, default=0.2)
    p.add_argument("--continuous_noise_scale", type=float, default=20.0)

    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--traj_num", type=int, default=1000)
    p.add_argument("--transition_num", type=int, default=10000)

    p.add_argument("--y_clip", type=float, default=1.0 - 1e-5)
    p.add_argument("--logdet_eps", type=float, default=1e-5)

    p.add_argument("--cuda", type=int, default=1)

    cli = p.parse_args()

    args = CollectArgs(
        checkpoint_path=cli.checkpoint_path,
        out_path=cli.out_path,
        use_ckpt_env=bool(cli.use_ckpt_env),

        env_name=cli.env_name,
        reward_type=cli.reward_type,
        episode_length=cli.episode_length,
        max_episode_steps=cli.max_episode_steps,
        start=cli.start,
        goal=cli.goal,
        goal_radius=cli.goal_radius,

        discrete_action=bool(cli.discrete_action),
        deterministic_base=bool(cli.deterministic_base),
        discrete_random_prob=cli.discrete_random_prob,
        continuous_noise_scale=cli.continuous_noise_scale,

        seed=cli.seed,
        traj_num=cli.traj_num,
        transition_num=cli.transition_num,

        y_clip=cli.y_clip,
        logdet_eps=cli.logdet_eps,

        cuda=bool(cli.cuda),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    set_seed(args.seed)

    ckpt = load_checkpoint(args.checkpoint_path, device)
    ckpt_cfg = ckpt.get("config", {}) or {}

    if args.use_ckpt_env and isinstance(ckpt_cfg, dict):
        merge_env_from_ckpt_cfg(args, ckpt_cfg)
        print(f"[ckpt] using env/action params from checkpoint config: {args.checkpoint_path}")

    actor, spec = build_actor_from_ckpt(ckpt, device)

    print("[collect] actor loaded.")
    print(f"[collect] obs_dim={spec['obs_dim']} act_dim={spec['act_dim']}")
    print(f"[collect] env_name={args.env_name} reward_type={args.reward_type} episode_length={args.episode_length}")
    print(f"[collect] start={args.start} goal={args.goal} goal_radius={args.goal_radius}")
    print(f"[collect] discrete_action={args.discrete_action} deterministic_base={args.deterministic_base}")
    print(f"[collect] transition_num={args.transition_num} seed={args.seed}")

    t0 = time.time()
    dataset = collect_dataset(actor, args, device)
    dt = time.time() - t0

    ensure_dir(os.path.dirname(args.out_path) if os.path.dirname(args.out_path) else ".")
    with open(args.out_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"[collect] saved: {args.out_path}")
    print(f"[collect] N={len(dataset['observations'])} transitions | took {dt:.2f}s")
