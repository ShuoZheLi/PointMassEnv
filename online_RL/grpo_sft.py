# grpo.py
# GRPO (Group Relative Policy Optimization) for PointMassEnv continuous actions
# with numerical stabilizations to prevent NaNs.
#
# Plug-and-play checkpointing update:
# - Saves a SINGLE .pt checkpoint that contains:
#   (1) actor weights, (2) ref_actor weights, (3) optimizer (optional),
#   (4) full config (Args as dict), (5) policy spec (obs_dim / act_dim / act_low / act_high),
#   (6) training counters (global_step/update_idx), and (7) RNG states (optional).
# - Adds "load_policy(...)" helper so others can do:
#       from grpo import load_policy
#       policy = load_policy("checkpoint.pt", device="cuda")
#       action = policy(obs, deterministic=True)
# - Adds CLI flags: --load_path, --resume, --eval_only, --use_ckpt_config
#
# Notes:
# - The Actor is refactored to be constructible from a saved "policy_spec" WITHOUT an env object.
# - For RESUME TRAINING with KL-to-ref, we also store and restore ref_actor.
# - If your checkpoint is from an older version that only saved actor.state_dict(),
#   you can still load it by setting --use_ckpt_config False and providing matching env args,
#   but you won’t have ref_actor/optimizer unless present.

import os, sys
import random
import time
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyrallis
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PointMassEnv import PointMassEnv  # noqa: E402

import imageio
import pickle


# -----------------------------
# Small utilities
# -----------------------------
def parse_vec2(s: str) -> np.ndarray:
    s = s.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    x, y = [float(v) for v in s.split(",")]
    return np.array([x, y], dtype=np.float32)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_np(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


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
    update_epochs: int = 1
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

    # run / logging
    # (kept for backward compatibility: you used this as run name)
    checkpoints_path: str = "checkpoints"

    # checkpointing
    save_checkpoints: bool = True
    checkpoint_every_steps: int = 10_000  # 0 disables periodic saves
    save_best: bool = True  # saves best eval return checkpoint as best.pt

    # loading / resuming
    load_path: str = ""      # path to a .pt checkpoint to load
    resume: bool = False     # if True, restores optimizer + counters + RNG
    eval_only: bool = False  # if True, just run eval and exit
    use_ckpt_config: bool = True  # if True, override Args (env+hyperparams) from checkpoint config

    # ref actor loading behavior (only used if checkpoint is missing ref_actor):
    # "from_ckpt" (default): load ref if present, else fallback to "reset_to_actor"
    # "reset_to_actor": set ref_actor = deepcopy(actor) after loading actor
    ref_mode: str = "from_ckpt"

    # env
    env_name: str = "FourRooms"
    reward_type: str = "sparse"
    discrete_action: bool = False
    episode_length: int = 120

    # start/goal from CLI (strings)
    start: str = "12.5,4.5"
    goal: str = "4.5,12.5"
    goal_radius: float = 0.8

    # eval
    eval_episodes: int = 1
    eval_deterministic: bool = True
    save_gifs: bool = True

    # -----------------------------
    # SFT / Behavior Cloning
    # -----------------------------
    sft_weight: float = 0.0
    sft_dataset_path: str = "hand_dataset.npy"
    sft_minibatch_size: int = 2048

    # UPDATE-BASED annealing:
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


def infer_policy_spec_from_env(env: gym.Env) -> Dict[str, Any]:
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("Only gym.spaces.Box action spaces are supported in this script.")
    obs_dim = int(np.array(env.observation_space.shape).prod())
    act_dim = int(np.prod(env.action_space.shape))
    act_low = to_np(env.action_space.low).reshape(-1)
    act_high = to_np(env.action_space.high).reshape(-1)
    if act_low.shape[0] != act_dim or act_high.shape[0] != act_dim:
        raise ValueError("Action space low/high shapes do not match act_dim.")
    return {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_low": act_low,
        "act_high": act_high,
        "hidden_dim": 256,
    }


# -----------------------------
# Actor (Squashed Gaussian) -- now env-free constructible
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

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor, y_clip: float, logdet_eps: float) -> torch.Tensor:
        y = (action - self.action_bias) / self.action_scale
        y = torch.clamp(y, -y_clip, y_clip)
        x = atanh(y)

        dist = self._dist(obs)
        logp = dist.log_prob(x)
        log_det = torch.log(self.action_scale) + torch.log(torch.clamp(1 - y.pow(2), min=logdet_eps))
        return (logp - log_det).sum(-1)


# -----------------------------
# Plug-and-play Policy wrapper
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


class Policy:
    """
    A tiny wrapper that makes a loaded checkpoint "plug and play".

    Example:
        policy = load_policy("path/to/checkpoint.pt", device="cuda")
        a = policy(obs, deterministic=True)  # obs: np.ndarray shape (obs_dim,)
    """

    def __init__(self, actor: Actor, device: torch.device, discrete_action: bool, y_clip: float, logdet_eps: float):
        self.actor = actor
        self.device = device
        self.discrete_action = bool(discrete_action)
        self.y_clip = float(y_clip)
        self.logdet_eps = float(logdet_eps)
        self.actor.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = to_np(obs).reshape(-1)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            a_t = self.actor.deterministic(obs_t)
        else:
            a_t, _ = self.actor.sample(obs_t, self.y_clip, self.logdet_eps)
        a = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if self.discrete_action:
            a = discrete_action_fn(a)[0].astype(np.float32)
        return a


# -----------------------------
# Checkpoint I/O
# -----------------------------
def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_random"] = None
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python_random" in state and state["python_random"] is not None:
        random.setstate(state["python_random"])
    if "numpy_random" in state and state["numpy_random"] is not None:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state and state["torch_random"] is not None:
        torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "torch_cuda_random" in state and state["torch_cuda_random"] is not None:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda_random"])
        except Exception:
            pass


def save_checkpoint(
    path: str,
    actor: Actor,
    ref_actor: Optional[Actor],
    optimizer: Optional[optim.Optimizer],
    args: Args,
    policy_spec: Dict[str, Any],
    global_step: int,
    env_step: int,
    update_idx: int,
    best_eval_return: float,
    include_optimizer: bool = True,
    include_rng: bool = True,
) -> None:
    payload: Dict[str, Any] = {
        "version": "grpo_pointmass_ckpt_v2",
        "time": time.time(),
        "config": asdict(args),
        "policy_spec": {
            "obs_dim": int(policy_spec["obs_dim"]),
            "act_dim": int(policy_spec["act_dim"]),
            "hidden_dim": int(policy_spec.get("hidden_dim", 256)),
            "act_low": to_np(policy_spec["act_low"]),
            "act_high": to_np(policy_spec["act_high"]),
        },
        "actor": actor.state_dict(),
        "ref_actor": ref_actor.state_dict() if ref_actor is not None else None,
        "global_step": int(global_step),
        "env_step": int(env_step),
        "update_idx": int(update_idx),
        "best_eval_return": float(best_eval_return),
    }
    if include_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    else:
        payload["optimizer"] = None

    if include_rng:
        payload["rng_state"] = get_rng_state()
    else:
        payload["rng_state"] = None

    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict.")
    return ckpt


def load_policy(checkpoint_path: str, device: str = "cpu") -> Policy:
    """
    Plug-and-play loader for others.

    Usage:
        from grpo import load_policy
        policy = load_policy(".../checkpoint.pt", device="cuda")
        action = policy(obs, deterministic=True)
    """
    dev = torch.device(device)
    ckpt = load_checkpoint(checkpoint_path, dev)

    spec = ckpt.get("policy_spec", None)
    if spec is None:
        raise KeyError("Checkpoint missing 'policy_spec'. Cannot construct Actor without env/spec.")

    # Convert act_low/high if saved as numpy arrays
    spec = dict(spec)
    spec["act_low"] = to_np(spec["act_low"])
    spec["act_high"] = to_np(spec["act_high"])

    actor = Actor.from_spec(spec).to(dev)
    actor.load_state_dict(ckpt["actor"], strict=True)
    actor.eval()

    cfg = ckpt.get("config", {}) or {}
    discrete_action = bool(cfg.get("discrete_action", False))
    y_clip = float(cfg.get("y_clip", 1.0 - 1e-5))
    logdet_eps = float(cfg.get("logdet_eps", 1e-5))

    return Policy(actor=actor, device=dev, discrete_action=discrete_action, y_clip=y_clip, logdet_eps=logdet_eps)


# -----------------------------
# Helpers
# -----------------------------
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
    save_gif: bool = True,
    deterministic: bool = True,
) -> Tuple[float, int, float]:
    """
    Returns: (avg_return_over_eval_episodes, avg_length, success_rate)
    """
    actor.eval()
    total_ret = 0.0
    total_len = 0
    total_succ = 0.0

    for ep in range(args.eval_episodes):
        env = PointMassEnv(
            start=start_np,
            goal=goal_np,
            goal_radius=args.goal_radius,
            env_name=args.env_name,
            reward_type=args.reward_type,
            episode_length=args.episode_length,
        )

        images = []
        obs, _ = env.reset(seed=args.seed + 12345 + ep)

        if save_gif:
            try:
                images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
            except Exception:
                images = []

        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_succ = 0.0

        while not done and ep_len < args.max_episode_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    a = actor.deterministic(obs_t).cpu().numpy()[0]
                else:
                    a, _ = actor.sample(obs_t, args.y_clip, args.logdet_eps)
                    a = a.cpu().numpy()[0]

            exec_a = a.astype(np.float32)
            if args.discrete_action:
                exec_a = discrete_action_fn(exec_a)[0].astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(exec_a)
            done = terminated or truncated

            if save_gif and len(images) > 0:
                try:
                    images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
                except Exception:
                    pass

            ep_ret += float(reward)
            ep_len += 1
            if done and isinstance(info, dict) and info.get("success", False):
                ep_succ = 1.0

        env.close()

        total_ret += ep_ret
        total_len += ep_len
        total_succ += ep_succ

        if save_gif and len(images) > 0 and args.save_gifs:
            ensure_dir(gif_dir)
            gif_path = os.path.join(gif_dir, f"{global_step}_ep{ep}.gif")
            try:
                imageio.mimsave(gif_path, images, fps=10)
                if args.track:
                    import wandb
                    wandb.log(
                        {f"eval/video_ep{ep}": wandb.Video(gif_path, fps=10, format="gif")},
                        step=global_step,
                    )
            except Exception as e:
                print(f"[eval] failed to save gif: {e}")

    actor.train()
    avg_ret = total_ret / float(max(1, args.eval_episodes))
    avg_len = int(round(total_len / float(max(1, args.eval_episodes))))
    succ_rate = total_succ / float(max(1, args.eval_episodes))

    print(f"[eval] step={global_step} avg_return={avg_ret:.2f} avg_len={avg_len} success_rate={succ_rate:.2f}")
    return avg_ret, avg_len, succ_rate


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


def merge_args_from_ckpt(args: Args, ckpt_cfg: Dict[str, Any]) -> None:
    """
    If args.use_ckpt_config == True, we overwrite most fields from checkpoint config,
    but keep runtime knobs (device/logging/loading switches) from current CLI.
    """
    # Keep these from current CLI run:
    keep = {
        "cuda",
        "track",
        "wandb_project_name",
        "wandb_entity",
        "capture_video",
        "load_path",
        "resume",
        "eval_only",
        "use_ckpt_config",
        "ref_mode",
        "save_checkpoints",
        "checkpoint_every_steps",
        "save_best",
        "eval_episodes",
        "eval_deterministic",
        "save_gifs",
        "checkpoints_path",
    }

    for k, v in ckpt_cfg.items():
        if hasattr(args, k) and (k not in keep):
            try:
                setattr(args, k, v)
            except Exception:
                pass


def build_ref_actor(actor: Actor, ckpt: Optional[Dict[str, Any]], device: torch.device, args: Args) -> Actor:
    if ckpt is not None and ckpt.get("ref_actor", None) is not None and args.ref_mode in {"from_ckpt", "reset_to_actor"}:
        ref_actor = deepcopy(actor).to(device)
        ref_actor.load_state_dict(ckpt["ref_actor"], strict=True)
    else:
        # fallback
        ref_actor = deepcopy(actor).to(device)

    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)
    return ref_actor


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = pyrallis.parse(config_class=Args)
    run_name = args.checkpoints_path

    # device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Optional: load checkpoint first (so we can override config before creating env)
    ckpt: Optional[Dict[str, Any]] = None
    if args.load_path:
        ckpt = load_checkpoint(args.load_path, device)
        ckpt_cfg = ckpt.get("config", {}) or {}
        if args.use_ckpt_config and isinstance(ckpt_cfg, dict):
            merge_args_from_ckpt(args, ckpt_cfg)
            print(f"[ckpt] using checkpoint config for env/hparams from: {args.load_path}")
        else:
            print(f"[ckpt] NOT using checkpoint config (use_ckpt_config=False). Loading weights only from: {args.load_path}")

    # parse start/goal AFTER potential config override
    start_np = parse_vec2(args.start)
    goal_np = parse_vec2(args.goal)

    # logging dirs
    log_dir = f"grpo_sft_anneal/{run_name}"
    gif_dir = os.path.join(log_dir, "gifs")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    ensure_dir(log_dir)
    ensure_dir(gif_dir)
    ensure_dir(ckpt_dir)

    writer = None
    if not args.eval_only:
        writer = SummaryWriter(log_dir)

    # tracking
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=asdict(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env
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

    # policy spec (prefer checkpoint spec if present)
    if ckpt is not None and ckpt.get("policy_spec", None) is not None:
        policy_spec = dict(ckpt["policy_spec"])
        policy_spec["act_low"] = to_np(policy_spec["act_low"])
        policy_spec["act_high"] = to_np(policy_spec["act_high"])
    else:
        policy_spec = infer_policy_spec_from_env(env)

    # build actor
    actor = Actor.from_spec(policy_spec).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

    # counters
    global_step = 0      # NOW: optimizer update step
    update_idx = 0       # keep as update counter (can mirror global_step)
    env_step = 0         # NEW: env interactions (for logging / traceability)
    best_eval_return = -1e18

    if ckpt is not None:
        if "actor" in ckpt and ckpt["actor"] is not None:
            actor.load_state_dict(ckpt["actor"], strict=True)
            print(f"[ckpt] loaded actor from: {args.load_path}")
        else:
            raise KeyError("Checkpoint missing 'actor' weights.")

        best_eval_return = float(ckpt.get("best_eval_return", best_eval_return))

        if args.resume:
            if ckpt.get("optimizer", None) is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("[ckpt] restored optimizer state.")
                except Exception as e:
                    print(f"[ckpt] WARNING: failed to restore optimizer state: {e}")

            ver = ckpt.get("version", "")

            if ver == "grpo_pointmass_ckpt_v1":
                # OLD checkpoints: global_step was env steps
                env_step = int(ckpt.get("global_step", 0))
                update_idx = int(ckpt.get("update_idx", 0))
                global_step = update_idx  # NEW semantics: updates
            else:
                # NEW checkpoints: global_step is updates, env_step stored separately
                global_step = int(ckpt.get("global_step", 0))
                update_idx = int(ckpt.get("update_idx", global_step))
                env_step = int(ckpt.get("env_step", 0))

            rng_state = ckpt.get("rng_state", None)
            if rng_state is not None:
                try:
                    set_rng_state(rng_state)
                    print("[ckpt] restored RNG state.")
                except Exception as e:
                    print(f"[ckpt] WARNING: failed to restore RNG state: {e}")

            print(f"[ckpt] resume=True -> global_step={global_step}, update_idx={update_idx}, env_step={env_step}")

    # ref actor (important for KL-to-ref)
    ref_actor = build_ref_actor(actor, ckpt, device, args)

    # ---- eval-only path ----
    if args.eval_only:
        avg_ret, avg_len, succ = eval_policy(
            actor=actor,
            global_step=global_step,
            gif_dir=gif_dir,
            args=args,
            device=device,
            start_np=start_np,
            goal_np=goal_np,
            save_gif=args.save_gifs,
            deterministic=args.eval_deterministic,
        )
        print(f"[eval_only] avg_return={avg_ret:.3f} avg_len={avg_len} success_rate={succ:.3f}")
        env.close()
        if writer is not None:
            writer.close()
        raise SystemExit(0)

    # ---- SFT dataset load (once) ----
    sft_enabled = (args.sft_weight > 0.0) or (args.sft_weight_end > 0.0)
    sft_obs_t = None
    sft_act_t = None
    sft_N = 0

    if sft_enabled:
        data = load_hand_dataset(args.sft_dataset_path)
        obs_np = np.asarray(data["observations"], dtype=np.float32)
        act_np = np.asarray(data["actions"], dtype=np.float32)

        if obs_np.ndim != 2 or obs_np.shape[1] != policy_spec["obs_dim"]:
            # Your env has obs_dim=2; we keep this check robust for future.
            raise ValueError(f"[SFT] Expected observations shape (N,{policy_spec['obs_dim']}), got {obs_np.shape}")
        if act_np.ndim != 2 or act_np.shape[1] != policy_spec["act_dim"]:
            raise ValueError(f"[SFT] Expected actions shape (N,{policy_spec['act_dim']}), got {act_np.shape}")

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
        writer.add_scalar("sft/weight_init", float(args.sft_weight), global_step)
        writer.add_scalar("sft/weight_end", float(args.sft_weight_end), global_step)
        writer.add_scalar("sft/anneal_updates", float(args.sft_anneal_updates), global_step)
        writer.add_scalar("sft/dataset_N", float(sft_N), global_step)

    # periodic saving bookkeeping
    last_ckpt_save_step = global_step

    start_time = time.time()

    while global_step < args.total_timesteps:
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

            # budget checks removed: we now count updates (global_step) and env interactions (env_step)

        env_step += batch_steps
        update_steps = len(obs_buf)
        if update_steps == 0:
            continue

        update_idx += 1
        global_step += 1   # ONE optimizer-update step per outer loop

        obs_t = torch.tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.stack(act_buf), dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(np.array(logp_old_buf), dtype=torch.float32, device=device)
        adv_t = torch.tensor(np.array(adv_buf), dtype=torch.float32, device=device)

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
            for start_i in range(0, update_steps, args.minibatch_size):
                mb_inds = inds[start_i : start_i + args.minibatch_size]
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

        ups = int(global_step / (time.time() - start_time + 1e-8))
        env_sps = int(env_step / (time.time() - start_time + 1e-8))
        writer.add_scalar("charts/UPS", ups, global_step)
        writer.add_scalar("charts/env_SPS", env_sps, global_step)
        writer.add_scalar("charts/batch_steps", batch_steps, global_step)
        writer.add_scalar("counters/env_step", env_step, global_step)
        writer.add_scalar("counters/env_steps_per_update", float(batch_steps), global_step)
        writer.add_scalar("losses/policy_loss", last_policy_loss, global_step)
        writer.add_scalar("losses/kl_est", last_kl, global_step)
        if sft_enabled:
            writer.add_scalar("losses/sft_nll", last_sft_nll, global_step)

        if len(batch_group_stats) > 0:
            writer.add_scalar("charts/group_return_mean", float(np.mean([x[0] for x in batch_group_stats])), global_step)
            writer.add_scalar("charts/group_return_std", float(np.mean([x[1] for x in batch_group_stats])), global_step)

        train_success_rate = float(np.mean(batch_successes)) if len(batch_successes) > 0 else 0.0
        writer.add_scalar("charts/train_success_rate", train_success_rate, global_step)

        writer.add_scalar("advantages/mean", adv_mean, global_step)
        writer.add_scalar("advantages/std", adv_std, global_step)
        writer.add_scalar("advantages/min", adv_min, global_step)
        writer.add_scalar("advantages/max", adv_max, global_step)
        writer.add_scalar("advantages/abs_mean", adv_abs_mean, global_step)
        writer.add_histogram("advantages/adv", adv_t.detach().cpu(), global_step)

        # --- DIRECT wandb logging (training) ---
        if args.track:
            import wandb
            wandb.log(
                {
                    "charts/UPS": ups,
                    "charts/env_SPS": env_sps,
                    "charts/batch_steps": batch_steps,
                    "counters/env_step": env_step,
                    "counters/env_steps_per_update": float(batch_steps),

                    "losses/policy_loss": last_policy_loss,
                    "losses/kl_est": last_kl,

                    "charts/train_success_rate": train_success_rate,
                    "advantages/mean": adv_mean,
                    "advantages/std": adv_std,
                    "advantages/min": adv_min,
                    "advantages/max": adv_max,
                    "advantages/abs_mean": adv_abs_mean,
                },
                step=global_step,
            )

            if sft_enabled:
                wandb.log(
                    {
                        "losses/sft_nll": last_sft_nll,
                        "sft/current_weight": float(cur_sft_weight),
                    },
                    step=global_step,
                )

            if len(batch_group_stats) > 0:
                wandb.log(
                    {
                        "charts/group_return_mean": float(np.mean([x[0] for x in batch_group_stats])),
                        "charts/group_return_std": float(np.mean([x[1] for x in batch_group_stats])),
                    },
                    step=global_step,
                )

        print(
            f"[update {update_idx}] update_step={global_step} env_step={env_step} batch_steps={batch_steps} "
            f"policy_loss={last_policy_loss:.4f} kl={last_kl:.4f} sft_nll={last_sft_nll:.4f} "
            f"sft_w={cur_sft_weight:.6f} train_succ={train_success_rate:.3f} "
            f"adv_mean={adv_mean:+.3f} adv_std={adv_std:.3f} adv_min={adv_min:+.3f} adv_max={adv_max:+.3f} UPS={ups} env_SPS={env_sps}"
        )

        # ---- periodic checkpoint save ----
        if args.save_checkpoints and args.checkpoint_every_steps > 0:
            if (global_step - last_ckpt_save_step) >= args.checkpoint_every_steps:
                ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    actor=actor,
                    ref_actor=ref_actor,
                    optimizer=optimizer,
                    args=args,
                    policy_spec=policy_spec,
                    global_step=global_step,
                    env_step=env_step,
                    update_idx=update_idx,
                    best_eval_return=best_eval_return,
                    include_optimizer=True,
                    include_rng=True,
                )
                last_ckpt_save_step = global_step
                print(f"[ckpt] saved: {ckpt_path}")

        # ---- eval & best checkpoint ----
        if global_step % 5 == 0:
            avg_ret, avg_len, succ = eval_policy(
                actor=actor,
                global_step=global_step,
                gif_dir=gif_dir,
                args=args,
                device=device,
                start_np=start_np,
                goal_np=goal_np,
                save_gif=args.save_gifs,
                deterministic=args.eval_deterministic,
            )
            writer.add_scalar("eval/return", avg_ret, global_step)
            writer.add_scalar("eval/length", avg_len, global_step)
            writer.add_scalar("eval/success_rate", succ, global_step)

            if args.track:
                import wandb
                wandb.log(
                    {
                        "eval/return": avg_ret,
                        "eval/length": avg_len,
                        "eval/success_rate": succ,
                    },
                    step=global_step,
                )

            if args.save_best and avg_ret > best_eval_return:
                best_eval_return = float(avg_ret)
                best_path = os.path.join(ckpt_dir, "best.pt")
                save_checkpoint(
                    path=best_path,
                    actor=actor,
                    ref_actor=ref_actor,
                    optimizer=optimizer,
                    args=args,
                    policy_spec=policy_spec,
                        global_step=global_step,
                        env_step=env_step,
                        update_idx=update_idx,
                    best_eval_return=best_eval_return,
                    include_optimizer=True,
                    include_rng=True,
                )
                print(f"[ckpt] new best avg_ret={best_eval_return:.3f} saved: {best_path}")

    # final save
    if args.save_checkpoints:
        final_path = os.path.join(ckpt_dir, "final.pt")
        save_checkpoint(
            path=final_path,
            actor=actor,
            ref_actor=ref_actor,
            optimizer=optimizer,
            args=args,
            policy_spec=policy_spec,
            global_step=global_step,
            env_step=env_step,
            update_idx=update_idx,
            best_eval_return=best_eval_return,
            include_optimizer=True,
            include_rng=True,
        )
        print(f"[ckpt] saved final: {final_path}")

    env.close()
    if writer is not None:
        writer.close()
