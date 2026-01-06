# grpo.py
# GRPO (Group Relative Policy Optimization) for PointMassEnv continuous actions
# with numerical stabilizations to prevent NaNs.
#
# UPDATED:
# 1) Adds SFT / behavior cloning term (weighted) using dataset like "hand_dataset.npy".
# 2) Accepts --start / --goal / --goal_radius from CLI and uses them everywhere (train env + eval env).
# 3) Adds linear annealing for SFT weight during training via flags (UPDATE-BASED).

import os, sys
import random
import time
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyrallis
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PointMassEnv import PointMassEnv
import imageio
import pickle


# -----------------------------
# Small utilities
# -----------------------------
def parse_vec2(s: str) -> np.ndarray:
    s = s.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    x, y = [float(v) for v in s.split(",")]
    return np.array([x, y], dtype=np.float32)


# -----------------------------
# Dataset loading (pickled dict saved as .npy in your collector)
# -----------------------------
def load_hand_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Expected: a dict with keys:
      observations: (N,2)
      actions: (N,2)
      ...
    Your collector saves via pickle into a .npy filename.
    """
    try:
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "observations" in data and "actions" in data:
            return data
    except Exception as e:
        print(f"[SFT] pickle load failed for {dataset_path}: {e}")

    try:
        obj = np.load(dataset_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
            data = obj.item()
            if "observations" in data and "actions" in data:
                return data
    except Exception as e:
        print(f"[SFT] np.load fallback failed for {dataset_path}: {e}")

    raise RuntimeError(
        f"[SFT] Could not load dataset from {dataset_path}. "
        f"Expected a pickled dict or an np.load-able dict with keys 'observations' and 'actions'."
    )


# -----------------------------
# Args
# -----------------------------
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = True
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    total_timesteps: int = 1_000_000
    gamma: float = 0.99

    # GRPO / PPO-style
    learning_rate: float = 1e-4
    clip_coef: float = 0.2
    kl_beta: float = 0.02
    update_epochs: int = 4
    minibatch_size: int = 2048
    max_grad_norm: float = 1.0

    group_size: int = 8
    groups_per_update: int = 8
    max_episode_steps: int = 512

    # numeric stability
    log_ratio_clip: float = 10.0
    log_ratio_ref_clip: float = 10.0
    y_clip: float = 1.0 - 1e-5
    logdet_eps: float = 1e-5

    # model / run
    save_model: bool = False
    checkpoints_path: str = "checkpoints"

    # env
    env_name: str = "FourRooms"
    reward_type: str = "sparse"
    discrete_action: bool = False
    episode_length: int = 120

    # start/goal from CLI (strings)
    start: str = "12.5,4.5"
    goal: str = "4.5,12.5"
    goal_radius: float = 0.8

    # -----------------------------
    # SFT / Behavior Cloning
    # -----------------------------
    sft_weight: float = 0.0
    sft_dataset_path: str = "hand_dataset.npy"
    sft_minibatch_size: int = 2048

    # UPDATE-BASED annealing:
    # Linear schedule: weight goes from sft_weight -> sft_weight_end over sft_anneal_updates POLICY UPDATES.
    # If sft_anneal_updates <= 0, weight stays constant at sft_weight.
    sft_weight_end: float = 0.0
    sft_anneal_updates: int = 0


# -----------------------------
# SFT weight schedule (UPDATE-BASED)
# -----------------------------
def get_sft_weight(args: Args, update_i0: int) -> float:
    """
    update_i0: 0-indexed update counter (0 for the 1st actual update).
    """
    w0 = float(args.sft_weight)
    w1 = float(args.sft_weight_end)
    U = int(args.sft_anneal_updates)
    if U <= 0:
        return w0
    t = float(update_i0) / float(U)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (1.0 - t) * w0 + t * w1


def make_env(
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    env_name: str,
    reward_type: str,
    episode_length: int,
    start: np.ndarray,
    goal: np.ndarray,
    goal_radius: float,
):
    def thunk():
        env = PointMassEnv(
            start=start,
            goal=goal,
            goal_radius=goal_radius,
            env_name=env_name,
            reward_type=reward_type,
            episode_length=episode_length,
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


# -----------------------------
# Actor (Squashed Gaussian)
# -----------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -5


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class Actor(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape).prod())
        act_dim = int(np.prod(env.action_space.shape))

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
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

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor, y_clip: float, logdet_eps: float) -> torch.Tensor:
        y = (action - self.action_bias) / self.action_scale
        y = torch.clamp(y, -y_clip, y_clip)
        x = atanh(y)

        dist = self._dist(obs)
        logp = dist.log_prob(x)
        log_det = torch.log(self.action_scale) + torch.log(torch.clamp(1 - y.pow(2), min=logdet_eps))
        return (logp - log_det).sum(-1)

    def save(self, model_path: str):
        torch.save(self.state_dict(), model_path)


# -----------------------------
# Helpers
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


def has_bad_params(model: nn.Module) -> bool:
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return True
    return False


def eval_policy(
    actor: Actor,
    global_step: int,
    gif_dir: str,
    args: Args,
    device: torch.device,
    start_np: np.ndarray,
    goal_np: np.ndarray,
):
    env = PointMassEnv(
        start=start_np,
        goal=goal_np,
        goal_radius=args.goal_radius,
        env_name=args.env_name,
        reward_type=args.reward_type,
        episode_length=args.episode_length,
    )

    actor.eval()
    images = []
    count_success = 0

    obs, _ = env.reset()
    images.append(np.moveaxis(np.transpose(env.render()), 0, -1))

    done = False
    ep_ret = 0.0
    ep_len = 0

    while not done and ep_len < args.max_episode_steps:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a = actor.deterministic(obs_t).cpu().numpy()[0]
        exec_a = a
        if args.discrete_action:
            exec_a = discrete_action_fn(exec_a)[0]

        obs, reward, terminated, truncated, info = env.step(exec_a)
        done = terminated or truncated
        images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
        ep_ret += float(reward)
        ep_len += 1
        if done and isinstance(info, dict) and info.get("success", False):
            count_success += 1

    actor.train()

    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f"{global_step}.gif")
    imageio.mimsave(gif_path, images, fps=10)

    if args.track:
        import wandb
        wandb.log({"policy_performance": wandb.Video(gif_path, fps=10, format="gif")})

    print(f"[eval] step={global_step} return={ep_ret:.2f} len={ep_len} success_rate={count_success/1.0:.2f}")
    return ep_ret, ep_len


def collect_group(
    env: gym.Env,
    policy_old: Actor,
    group_seed: int,
    group_size: int,
    device: torch.device,
    args: Args,
) -> Tuple[List[float], List[List[Tuple[np.ndarray, np.ndarray, float]]], int, List[float]]:
    returns = []
    trajs = []
    total_steps = 0
    successes = []

    for _ in range(group_size):
        obs, _ = env.reset(seed=group_seed)
        done = False
        t = 0
        disc = 1.0
        ret = 0.0
        traj = []
        ep_success = 0.0

        while not done and t < args.max_episode_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a_cont, _ = policy_old.sample(obs_t, args.y_clip, args.logdet_eps)

            a_np = a_cont.cpu().numpy()[0].astype(np.float32)
            exec_a = a_np
            if args.discrete_action:
                exec_a = discrete_action_fn(exec_a)[0].astype(np.float32)

            exec_a_t = torch.tensor(exec_a, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logp_old = policy_old.log_prob(obs_t, exec_a_t, args.y_clip, args.logdet_eps).cpu().item()

            next_obs, reward, terminated, truncated, info = env.step(exec_a)
            done = terminated or truncated

            if done and isinstance(info, dict):
                if bool(info.get("success", False)) and bool(info.get("valid", False)):
                    ep_success = 1.0

            traj.append((obs.astype(np.float32), exec_a.astype(np.float32), float(logp_old)))

            ret += disc * float(reward)
            obs = next_obs
            disc *= args.gamma
            t += 1
            total_steps += 1

        returns.append(ret)
        trajs.append(traj)
        successes.append(ep_success)

    return returns, trajs, total_steps, successes


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = pyrallis.parse(config_class=Args)
    run_name = args.checkpoints_path

    start_np = parse_vec2(args.start)
    goal_np = parse_vec2(args.goal)

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    log_dir = f"grpo_sft_anneal/{run_name}"
    writer = SummaryWriter(log_dir)
    gif_dir = os.path.join(log_dir, "gifs")
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(
        seed=args.seed,
        idx=0,
        capture_video=args.capture_video,
        run_name=run_name,
        env_name=args.env_name,
        reward_type=args.reward_type,
        episode_length=args.episode_length,
        start=start_np,
        goal=goal_np,
        goal_radius=args.goal_radius,
    )()

    assert isinstance(env.action_space, gym.spaces.Box), "Only continuous action space supported here."

    actor = Actor(env).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

    ref_actor = deepcopy(actor).to(device)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    # Load SFT dataset (once)
    sft_enabled = (args.sft_weight > 0.0) or (args.sft_weight_end > 0.0)
    sft_obs_t = None
    sft_act_t = None
    sft_N = 0

    if sft_enabled:
        data = load_hand_dataset(args.sft_dataset_path)
        obs_np = np.asarray(data["observations"], dtype=np.float32)
        act_np = np.asarray(data["actions"], dtype=np.float32)

        if obs_np.ndim != 2 or obs_np.shape[1] != 2:
            raise ValueError(f"[SFT] Expected observations shape (N,2), got {obs_np.shape}")
        if act_np.ndim != 2 or act_np.shape[1] != 2:
            raise ValueError(f"[SFT] Expected actions shape (N,2), got {act_np.shape}")

        if args.discrete_action:
            act_np = discrete_action_fn(act_np).astype(np.float32)

        sft_obs_t = torch.tensor(obs_np, dtype=torch.float32, device=torch.device("cpu"))
        sft_act_t = torch.tensor(act_np, dtype=torch.float32, device=torch.device("cpu"))
        sft_N = obs_np.shape[0]

        print(
            f"[SFT] enabled: path={args.sft_dataset_path} N={sft_N} "
            f"sft_weight={args.sft_weight} -> {args.sft_weight_end} over {args.sft_anneal_updates} updates "
            f"sft_mb={args.sft_minibatch_size}"
        )
        writer.add_scalar("sft/weight_init", float(args.sft_weight), 0)
        writer.add_scalar("sft/weight_end", float(args.sft_weight_end), 0)
        writer.add_scalar("sft/anneal_updates", float(args.sft_anneal_updates), 0)
        writer.add_scalar("sft/dataset_N", float(sft_N), 0)

    start_time = time.time()
    global_step = 0
    update_idx = 0  # counts ACTUAL successful updates

    while global_step < args.total_timesteps:
        # NOTE: don't increment update_idx until we know we actually have data (update_steps>0)

        old_actor = deepcopy(actor).to(device)
        old_actor.eval()
        for p in old_actor.parameters():
            p.requires_grad_(False)

        obs_buf: List[np.ndarray] = []
        act_buf: List[np.ndarray] = []
        logp_old_buf: List[float] = []
        adv_buf: List[float] = []

        batch_steps = 0
        batch_group_stats = []
        batch_successes = []

        # Use a "proposed" update number for seeding (so seeds still advance even if a rare empty batch happens)
        proposed_update_num = update_idx + 1

        for g in range(args.groups_per_update):
            group_seed = args.seed + proposed_update_num * 10_000 + g
            returns, trajs, steps, successes = collect_group(env, old_actor, group_seed, args.group_size, device, args)
            batch_successes.extend(successes)

            batch_steps += steps
            batch_group_stats.append((float(np.mean(returns)), float(np.std(returns))))

            r = np.array(returns, dtype=np.float32)
            r_mean = float(r.mean())
            r_std = float(r.std() + 1e-8)
            adv_scalars = (r - r_mean) / r_std

            for i in range(args.group_size):
                adv_i = float(adv_scalars[i])
                for (o, a, lp_old) in trajs[i]:
                    obs_buf.append(o)
                    act_buf.append(a)
                    logp_old_buf.append(lp_old)
                    adv_buf.append(adv_i)

            if global_step + batch_steps >= args.total_timesteps:
                break

        global_step += batch_steps
        update_steps = len(obs_buf)
        if update_steps == 0:
            continue

        # Now we confirm this is a real update
        update_idx += 1

        obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.stack(act_buf), dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(np.array(logp_old_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(np.array(adv_buf), dtype=torch.float32, device=device)

        # UPDATE-BASED: first update uses update_i0=0
        cur_sft_weight = get_sft_weight(args, update_idx - 1)
        if sft_enabled:
            writer.add_scalar("sft/current_weight", float(cur_sft_weight), global_step)

        with torch.no_grad():
            adv_mean = float(adv_t.mean().detach().cpu().item())
            adv_std = float(adv_t.std(unbiased=False).detach().cpu().item())
            adv_min = float(adv_t.min().detach().cpu().item())
            adv_max = float(adv_t.max().detach().cpu().item())
            adv_abs_mean = float(adv_t.abs().mean().detach().cpu().item())

        with torch.no_grad():
            logp_ref_t = ref_actor.log_prob(obs_t, act_t, args.y_clip, args.logdet_eps)

        actor.train()
        inds = np.arange(update_steps)

        last_policy_loss = 0.0
        last_kl = 0.0
        last_sft_nll = 0.0

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, update_steps, args.minibatch_size):
                mb_inds = inds[start : start + args.minibatch_size]
                mb_obs = obs_t[mb_inds]
                mb_act = act_t[mb_inds]
                mb_logp_old = logp_old_t[mb_inds]
                mb_adv = adv_t[mb_inds]
                mb_logp_ref = logp_ref_t[mb_inds]

                logp = actor.log_prob(mb_obs, mb_act, args.y_clip, args.logdet_eps)
                if torch.isnan(logp).any() or torch.isinf(logp).any():
                    continue

                log_ratio = torch.clamp(logp - mb_logp_old, -args.log_ratio_clip, args.log_ratio_clip)
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_adv
                clipped_obj = torch.min(surr1, surr2).mean()

                log_ratio_ref = torch.clamp(mb_logp_ref - logp, -args.log_ratio_ref_clip, args.log_ratio_ref_clip)
                ratio_ref = torch.exp(log_ratio_ref)
                kl = (ratio_ref - log_ratio_ref - 1.0).mean()

                sft_obj = torch.tensor(0.0, device=device)
                if sft_enabled and sft_N > 0 and cur_sft_weight > 0.0:
                    sft_bs = min(args.sft_minibatch_size, sft_N)
                    sft_idx = np.random.randint(0, sft_N, size=sft_bs)
                    sft_obs_mb = sft_obs_t[sft_idx].to(device)
                    sft_act_mb = sft_act_t[sft_idx].to(device)

                    sft_logp = actor.log_prob(sft_obs_mb, sft_act_mb, args.y_clip, args.logdet_eps)
                    if not (torch.isnan(sft_logp).any() or torch.isinf(sft_logp).any()):
                        sft_obj = sft_logp.mean()

                objective = clipped_obj + (cur_sft_weight * sft_obj) - (args.kl_beta * kl)
                loss = -objective

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optimizer.step()

                if has_bad_params(actor):
                    actor.load_state_dict(old_actor.state_dict())
                    break

                last_policy_loss = float((-clipped_obj).detach().cpu().item())
                last_kl = float(kl.detach().cpu().item())
                if sft_enabled:
                    last_sft_nll = float((-(sft_obj.detach())).cpu().item())

        sps = int(global_step / (time.time() - start_time + 1e-8))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/batch_steps", batch_steps, global_step)
        writer.add_scalar("losses/policy_loss", last_policy_loss, global_step)
        writer.add_scalar("losses/kl_est", last_kl, global_step)
        if sft_enabled:
            writer.add_scalar("losses/sft_nll", last_sft_nll, global_step)

        if len(batch_group_stats) > 0:
            writer.add_scalar("charts/group_return_mean", float(np.mean([x[0] for x in batch_group_stats])), global_step)
            writer.add_scalar("charts/group_return_std", float(np.mean([x[1] for x in batch_group_stats])), global_step)

        if len(batch_successes) > 0:
            train_success_rate = float(np.mean(batch_successes))
            writer.add_scalar("charts/train_success_rate", train_success_rate, global_step)
        else:
            train_success_rate = 0.0

        writer.add_scalar("advantages/mean", adv_mean, global_step)
        writer.add_scalar("advantages/std", adv_std, global_step)
        writer.add_scalar("advantages/min", adv_min, global_step)
        writer.add_scalar("advantages/max", adv_max, global_step)
        writer.add_scalar("advantages/abs_mean", adv_abs_mean, global_step)
        writer.add_histogram("advantages/adv", adv_t.detach().cpu(), global_step)

        print(
            f"[update {update_idx}] step={global_step} batch_steps={batch_steps} "
            f"policy_loss={last_policy_loss:.4f} kl={last_kl:.4f} sft_nll={last_sft_nll:.4f} "
            f"sft_w={cur_sft_weight:.6f} "
            f"train_succ={train_success_rate:.3f} "
            f"adv_mean={adv_mean:+.3f} adv_std={adv_std:.3f} adv_min={adv_min:+.3f} adv_max={adv_max:+.3f} "
            f"SPS={sps}"
        )

        if global_step % 10_000 < batch_steps:
            eval_ret, eval_len = eval_policy(actor, global_step, gif_dir, args, device, start_np, goal_np)
            writer.add_scalar("eval/return", eval_ret, global_step)
            writer.add_scalar("eval/length", eval_len, global_step)
            if args.save_model:
                actor.save(os.path.join(model_dir, f"{global_step}_actor.pth"))

    env.close()
    writer.close()
