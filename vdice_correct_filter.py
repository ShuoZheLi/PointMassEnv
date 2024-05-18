# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
# import suboptimal_offline_datasets
from torch import autograd
from utils import TwinQ, ValueFunction, GaussianPolicy, DeterministicPolicy, ReplayBuffer, MLP, TwinV, soft_update, set_seed, compute_mean_std, eval_actor, return_reward_range, modify_reward, normalize_states, wrap_env
from toy_env import PointMassEnv, WALLS


TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainConfig:
    # Experiment
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    save_freq: int = int(1e3)  # How often (time steps) save the model
    update_freq: int = int(1.5e5)  # How often (time steps) we update the model
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(6e6)  # Max time steps to run environment
    load_semi_model: str = None  # Model load file name, "" doesn't load
    load_true_model: str = None  # Model load file name, "" doesn't load
    load_id_dataset: str = None  # Model load file name, "" doesn't load
    # VDICE
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    # batch_size: int = 512  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    
    
    # iql_deterministic: bool = False  # Use deterministic actor

    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network


    # Wandb logging
    # project: str = "hm_odice"
    project: str = "test_hopper"
    env_1: str = "antmaze-umaze-v2"  # OpenAI gym environment name
    env_2: str = "antmaze-umaze-v2"  # OpenAI gym environment name
    # env: str = "hopper-medium-v2"  # OpenAI gym environment name
    # group: str = "VDICE-D4RL"
    seed: int = 100

    
    
    vdice_lambda: float = 0.6
    vdice_eta: float = 4.0
    device: str = "cuda"
    alg: str = "100_0st"
    # checkpoints_path: Optional[str] = "model/"
    checkpoints_path: Optional[str] = "test"
    vdice_type: Optional[str] = "semi"
    
    combine: bool = False
    filter_dataset: bool = False
    w_threshold: float = 0

    load_true_dice: Optional[str] = ""
    load_vdice: Optional[str] = ""
    # load_true_dice: Optional[str] = "/robodata/corl/iql_reverse_kl_preload_value/algorithms/dice/own_dataset_trueDice_hopper-random-expert-0_1/100_0_95_hopper-random-expert-0_1-v2/100_0st-hopper-random-expert-0.1-v2-80df24f5/checkpoint_999999.pt"
    # load_vdice: Optional[str] = "/robodata/corl/iql_reverse_kl_preload_value/algorithms/dice/truedice_combine_vdice/100_0_5_truedice_combine_vdice/100_0st-hopper-random-expert-0.1-v2-3e32d6f2/checkpoint_999999.pt"

    semi_vdice_lambda: float = 0.55
    # semi_vdice_lambda: float = 0.4237
    true_vdice_lambda: float = 0.99

    semi_lambda_delta: float = 0
    true_lambda_delta: float = 0

    true_alpha: float = 100.0
    semi_eta: float = 1.0

    bc_max_timesteps: int = 3e5
    bc_policy_step: int = 0
    iql_deterministic: bool = False  # Use deterministic actor

    policy_ratio: bool = False
    state_ratio: bool = False
    or_ration: bool = False
    and_ratio: bool = False

    expert_num: int = 1e5
    layernorm: bool = False
    hidden_dim: int = 256
    

    def __post_init__(self):

        self.name = f"{self.alg}-{self.env_1}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)



def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        # group=config["group"],
        # name=config["env"] + '_' + config["alg"],
        name=config["alg"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()
    wandb.save(os.path.abspath(__file__))


    # define our custom x axis metric
    wandb.define_metric("bc_step")
    # set all other train/ metrics to use this step
    wandb.define_metric("bc_train/*", step_metric="bc_step")

    wandb.define_metric("vdice_step")
    wandb.define_metric("vdice_train/*", step_metric="vdice_step")


def frenchel_dual(name, x):
    if name == "Reverse_KL":
        return torch.exp(x - 1)
    elif name == "Pearson_square_chi":
        return torch.max(x + x**2 / 4, torch.zeros_like(x))
    elif name == "Smoothed_square_chi":
        omega_star = torch.max(x / 2 + 1, torch.zeros_like(x))
        return x * omega_star - (omega_star - 1)**2
    else:
        raise ValueError("Unknown policy f name")

def frenchel_dual_prime(name, x):
    if name == "Smoothed_square_chi":
        omega_star = torch.max(x / 2 + 1, torch.zeros_like(x))
        return omega_star
    else:
        raise ValueError("Unknown policy f name")

def f_prime_inverse(name, x, temperatrue=3.0):
    if name == "Reverse_KL":
        return torch.exp(x * temperatrue)
    elif "square_chi" in name:
        return torch.max(x, torch.zeros_like(x))
    else:
        raise ValueError("Unknown policy f name")


class VDICE:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        true_v_network: nn.Module,
        true_v_optimizer: torch.optim.Optimizer,
        semi_v_network: nn.Module,
        semi_v_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        mu_network: nn.Module,
        mu_optimizer: torch.optim.Optimizer,
        U_network: nn.Module,
        U_optimizer: torch.optim.Optimizer,
        vdice_type: str = "semi",
        semi_lambda: float = 0.7,
        semi_eta: float = 0.7,
        true_alpha: float = 0.1,
        f_name: str = 'Smoothed_square_chi',
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.true_v = true_v_network
        self.semi_v = semi_v_network
        self.q = q_network
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(device)
        self.mu = mu_network
        self.U = U_network
        self.actor = actor
        self.true_v_optimizer = true_v_optimizer
        self.semi_v_optimizer = semi_v_optimizer
        self.q_optimizer = q_optimizer
        self.mu_optimizer = mu_optimizer
        self.U_optimizer = U_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.vdice_type = vdice_type
        self.vdice_lambda = semi_lambda
        self.semi_eta = semi_eta
        self.true_alpha = true_alpha
        self.f_name = f_name
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        init_observations: torch.Tensor,
        log_dict: Dict,
    ):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        semi_v = self.semi_v(observations)
        adv = target_q - semi_v
        semi_linear_loss = (1 - self.vdice_lambda) * semi_v
        forward_dual_loss = self.vdice_lambda * frenchel_dual(self.f_name, adv)
        semi_v_loss = semi_linear_loss + forward_dual_loss
        u = self.U(observations)

        semi_v_loss = torch.mean(semi_v_loss)
        semi_pi_residual = adv.clone().detach()

        self.semi_v_optimizer.zero_grad()
        semi_v_loss.backward()
        self.semi_v_optimizer.step()

        true_v = self.true_v(observations)
        true_v_next = self.true_v(next_observations)
        true_residual = rewards + (1.0 - terminals.float()) * self.discount * true_v_next - true_v
        true_residual = true_residual / self.true_alpha
        true_dual_loss = torch.mean(frenchel_dual(self.f_name, true_residual))
        true_linear_loss = (1 - self.discount) * torch.mean(self.true_v(init_observations))
        true_v_loss = true_linear_loss + true_dual_loss
        true_pi_residual = true_residual.clone().detach()

        self.true_v_optimizer.zero_grad()
        true_v_loss.backward()
        self.true_v_optimizer.step()

        log_dict["vdice_train/semi_v_value"] = semi_v.mean().item()
        log_dict["vdice_train/true_v_value"] = true_v.mean().item()

        # Update target Q network
        # soft_update(self.v_target, self.v, self.tau)
        return semi_pi_residual, true_pi_residual

    def _update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        with torch.no_grad():
            next_v = self.semi_v(next_observations)
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v
        qs = self.q.both(observations, actions)
        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        u = self.U(observations)
        s_weight = f_prime_inverse(self.f_name, u)
        # q_loss = sum(torch.mean(s_weight * (q-targets)**2) for q in qs) / len(qs)
        q_loss = sum(torch.mean((q-targets)**2) for q in qs) / len(qs)
        log_dict["vdice_train/q_loss"] = q_loss.item()
        log_dict["vdice_train/q_value"] = qs[0].mean().item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.q, self.tau)

    def _update_U(
        self,
        semi_pi_residual: torch.Tensor,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        u = self.U(observations)
        mu = self.mu(observations)
        mu_next = self.mu(next_observations)
        mu_residual = (1.0 - terminals.float()) * self.discount * mu_next - mu
        a_weight = f_prime_inverse(self.f_name, semi_pi_residual)
        u_target = a_weight * mu_residual
        u_weight = (a_weight * frenchel_dual_prime(self.f_name, u)).clone().detach()

        u_loss = F.mse_loss(u, u_target)

        self.U_optimizer.zero_grad()
        u_loss.backward()
        self.U_optimizer.step()

        log_dict["vdice_train/a_weight"] = a_weight.mean().item()
        log_dict["vdice_train/U_loss"] = u_loss.item()
        log_dict["vdice_train/U_value"] = u.mean().item()

        return u_weight

    def _update_mu(
        self,
        u_weight: torch.Tensor,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
        init_observations: torch.Tensor,
        log_dict: Dict,
    ):
        mu = self.mu(observations)
        mu_next = self.mu(next_observations)
        mu_residual = (1.0 - terminals.float()) * self.discount * mu_next - mu
        mu_dual_loss = torch.mean(u_weight * frenchel_dual(self.f_name, mu_residual))
        mu_linear_loss = (1 - self.discount) * torch.mean(self.mu(init_observations))
        mu_loss = mu_linear_loss + mu_dual_loss

        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        log_dict["vdice_train/mu_loss"] = mu_loss.item()
        log_dict["vdice_train/mu_value"] = mu.mean().item()


    def _update_policy(
        self,
        semi_pi_residual: torch.Tensor,
        true_pi_residual: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        flags: torch.Tensor,
        log_dict: Dict,
    ):
        u = self.U(observations)
        s_weight = f_prime_inverse(self.f_name, u)
        a_weight = f_prime_inverse(self.f_name, semi_pi_residual)
        semi_sa_weight = s_weight * a_weight
        true_sa_weight = f_prime_inverse(self.f_name, true_pi_residual)
        # policy_out = self.actor(observations)
        # bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        # policy_loss = torch.mean(a_weight * bc_losses)
        # policy_loss = torch.sum(flags.squeeze() * bc_losses) / torch.sum(flags.squeeze() == 1)

        # flags = flags.squeeze()
        # log_dict["vdice_train/semi_a_sim"] = (((a_weight * flags) > 0).sum() / (a_weight > 0).sum()).item()
        # log_dict["vdice_train/semi_s_sim"] = (((s_weight * flags) > 0).sum() / (s_weight > 0).sum()).item()
        # log_dict["vdice_train/semi_sa_sim"] = (((semi_sa_weight * flags) > 0).sum() / (semi_sa_weight > 0).sum()).item()
        # log_dict["vdice_train/true_sa_sim"] = (((true_sa_weight * flags) > 0).sum() / (true_sa_weight > 0).sum()).item()

        # log_dict["vdice_train/semi_a_num"] = (((a_weight * flags) > 0).sum()).item()
        # log_dict["vdice_train/semi_s_num"] = (((s_weight * flags) > 0).sum()).item()
        # log_dict["vdice_train/semi_sa_num"] = (((semi_sa_weight * flags) > 0).sum()).item()
        # log_dict["vdice_train/true_sa_num"] = (((true_sa_weight * flags) > 0).sum()).item()

        # log_dict["actor_loss"] = policy_loss.item()
        log_dict["vdice_train/s_weight"] = s_weight.mean().item()
        log_dict["vdice_train/a_weight"] = a_weight.mean().item()
        log_dict["vdice_train/semi_sa_weight"] = s_weight.mean().item()
        log_dict["vdice_train/true_sa_weight"] = true_sa_weight.mean().item()



        # self.actor_optimizer.zero_grad()
        # policy_loss.backward()
        # self.actor_optimizer.step()
        # self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1

        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            # flags,
        ) = batch
        flags = torch.ones_like(rewards)
        
        log_dict = {}

        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update V function
        semi_residual, true_residual = self._update_v(observations, actions, next_observations, rewards, dones, observations, log_dict)
        # Update Q function
        self._update_q(observations, actions, next_observations, rewards, dones, log_dict)
        # Update U function
        u_weight = self._update_U(semi_residual, observations, next_observations, dones, log_dict)
        # Update Mu function
        self._update_mu(u_weight, observations, next_observations, dones, observations, log_dict)

        # Update actor
        self._update_policy(semi_residual, true_residual, observations, actions, flags, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "semi_v": self.semi_v.state_dict(),
            "semi_v_optimizer": self.semi_v_optimizer.state_dict(),
            "true_v": self.true_v.state_dict(),
            "true_v_optimizer": self.true_v_optimizer.state_dict(),
            "mu": self.mu.state_dict(),
            "mu_optimizer": self.mu_optimizer.state_dict(),
            "U": self.U.state_dict(),
            "U_optimizer": self.U_optimizer.state_dict(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.semi_v.load_state_dict(state_dict["semi_v"])
        self.semi_v_optimizer.load_state_dict(state_dict["semi_v_optimizer"])
        self.true_v.load_state_dict(state_dict["true_v"])
        self.true_v_optimizer.load_state_dict(state_dict["true_v_optimizer"])
        self.mu.load_state_dict(state_dict["mu"])
        self.mu_optimizer.load_state_dict(state_dict["mu_optimizer"])
        self.U.load_state_dict(state_dict["U"])
        self.U_optimizer.load_state_dict(state_dict["U_optimizer"])
        self.q.load_state_dict(state_dict["q"])
        self.q_target.load_state_dict(state_dict["q_target"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = state_dict["total_it"]

class BcLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        max_steps: int = 1000000,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.discount = discount
        self.total_it = 0
        self.device = device

    
    def _update_policy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(bc_losses)
        log_dict["bc_train/actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            experts,
        ) = batch
        log_dict = {}

        self._update_policy(observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        self.total_it = state_dict["total_it"]


def odice_result(dataset, trainer, config):
    weights = []
    for i in range(0,len(dataset["observations"]),8192):
        with torch.no_grad():
            target_v_next = trainer.v_target(torch.FloatTensor(dataset["next_observations"][i:min(i+8192,len(dataset["next_observations"]))]).to(config.device)).cpu().numpy()


            v = trainer.v(torch.FloatTensor(dataset["observations"][i:min(i+8192,len(dataset["observations"]))]).to(config.device)).detach().cpu().numpy()
            forward_residual = dataset["rewards"][i:min(i+8192,len(dataset["next_observations"]))] + (1.0 - dataset["terminals"][i:min(i+8192,len(dataset["next_observations"]))].astype(float)) * 0.99 * target_v_next - v
            v_loss_weight = f_prime_inverse(trainer.f_name, torch.FloatTensor(forward_residual))
            weights.append(v_loss_weight.mean(0))

    weights = np.hstack(weights)
    weights = weights > 0
    return weights

def truedice_result(dataset, trainer, config):
    weights = []
    for i in range(0,len(dataset["observations"]),8192):
        with torch.no_grad():
            v = trainer.v.both(torch.FloatTensor(dataset["observations"][i:min(i+8192,len(dataset["observations"]))]).to(config.device))
            v_next = trainer.v.both(torch.FloatTensor(dataset["next_observations"][i:min(i+8192,len(dataset["next_observations"]))]).to(config.device))
            rewards = torch.FloatTensor(dataset["rewards"][i:min(i+8192,len(dataset["rewards"]))]).to(config.device)
            terminals = torch.FloatTensor((1.0 - dataset["terminals"][i:min(i+8192,len(dataset["terminals"]))].astype(float))).to(config.device)

            pi_residual = rewards + terminals * config.discount * v_next - v
            weights.append(pi_residual.mean(0).cpu().numpy())
    
    weights = np.hstack(weights)
    return weights



def semidice_result(dataset, trainer, config):
    weights = []
    for i in range(0,len(dataset["observations"]),8192):
        with torch.no_grad():
            v = trainer.v.both(torch.FloatTensor(dataset["observations"][i:min(i+8192,len(dataset["observations"]))]).to(config.device))
            target_v_next = trainer.v_target(torch.FloatTensor(dataset["next_observations"][i:min(i+8192,len(dataset["next_observations"]))]).to(config.device))
            rewards = torch.FloatTensor(dataset["rewards"][i:min(i+8192,len(dataset["rewards"]))]).to(config.device)
            terminals = torch.FloatTensor((1.0 - dataset["terminals"][i:min(i+8192,len(dataset["terminals"]))].astype(float))).to(config.device)

            pi_residual = rewards + terminals * config.discount * target_v_next - v
            pi_residual = f_prime_inverse(trainer.f_name, pi_residual)
            weights.append(pi_residual.mean(0).cpu().numpy())
    
    weights = np.hstack(weights)
    return weights

def semidice_result_no_vtarget(dataset, trainer, config):

    a_weights = []
    s_weights = []
    semi_vs = []
    for i in range(0,len(dataset["observations"]),8192):

            # v_s.append(v.mean(0).cpu().numpy())

        with torch.no_grad():
            observations = torch.FloatTensor(dataset["observations"][i:min(i+8192,len(dataset["observations"]))]).to(config.device)
            actions = torch.FloatTensor(dataset["actions"][i:min(i+8192,len(dataset["actions"]))]).to(config.device)
            
            target_q = trainer.q_target(observations, actions)
            semi_v = trainer.semi_v(observations)
            adv = (target_q - semi_v)
            a_weight = f_prime_inverse(trainer.f_name, adv)

            u = trainer.U(observations)
            s_weight = f_prime_inverse(trainer.f_name, u)

            a_weights.append(a_weight.cpu().numpy())
            s_weights.append(s_weight.cpu().numpy())
            semi_vs.append(semi_v.cpu().numpy())

    if len(a_weights) == 0:
        a_weights.append([0])
    if len(s_weights) == 0:
        s_weights.append([0])
    if len(semi_vs) == 0:
        semi_vs.append([0])

    a_weights = np.hstack(a_weights)
    s_weights = np.hstack(s_weights)
    semi_vs = np.hstack(semi_vs)
    return a_weights, s_weights, semi_vs


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    timeout_ = []
    task_horizon = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        timeout_bool = bool(dataset['timeouts'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        timeout_.append(timeout_bool)
        task_horizon.append(episode_step)
        episode_step += 1
    

    # add in return for each episode
    return_list = [0]
    length = [0]
    for i in range(len(done_)):
        return_list[-1] += reward_[i]
        length[-1] += 1
        if done_[i] or timeout_[i]:
            return_list.append(0)
            length.append(0)

    count = 0
    data_return_list = [0] * len(done_)
    for i in range(len(done_)):
        data_return_list[i] = return_list[count]
        if done_[i] or timeout_[i]:
            count +=1
    
    data_return_list = env.get_normalized_score(np.array(data_return_list)) * 100.0
    data_return_list = np.array(data_return_list)


    epi_obs = []
    epi_n_obs = []
    epi_terminals = []
    epi_rewards = []
    epi_returns = []
    epi_actions = []
    obs = []
    n_obs = []
    terminals = []
    rewards = []
    actions = []
    # task_horizon = []
    task_step = 0
    for i in range(len(done_)):
        obs.append(obs_[i])
        n_obs.append(next_obs_[i])
        terminals.append(done_[i])
        rewards.append(reward_[i])
        actions.append(action_[i])
        # task_horizon.append(task_step)
        task_step += 1
        if done_[i] or timeout_[i]:
            epi_obs.append(np.array(obs))
            epi_n_obs.append(np.array(n_obs))
            epi_terminals.append(np.array(terminals))
            epi_rewards.append(np.array(rewards))
            epi_returns.append(data_return_list[i])
            epi_actions.append(np.array(actions))
            obs = []
            n_obs = []
            terminals = []
            rewards = []
            actions = []
            task_step = 0

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'timeouts': np.array(timeout_),
        'returns': data_return_list,
        'epi_obs': np.array(epi_obs, dtype=object),
        'epi_n_obs': np.array(epi_n_obs, dtype=object),
        'epi_terminals': np.array(epi_terminals, dtype=object),
        'epi_rewards': np.array(epi_rewards, dtype=object),
        'epi_returns': np.array(epi_returns, dtype=object),
        'epi_actions': np.array(epi_actions, dtype=object),
        'task_horizon':np.array(task_horizon, dtype=object),
    }

def trainer_init(config: TrainConfig, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        os.makedirs(config.checkpoints_path+"_semi", exist_ok=True)
        os.makedirs(config.checkpoints_path+"_or_ratio_id", exist_ok=True)
        os.makedirs(config.checkpoints_path+"_and_ratio_id", exist_ok=True)
        os.makedirs(config.checkpoints_path+"_action_ratio_id", exist_ok=True)
        os.makedirs(config.checkpoints_path+"_state_ratio_id", exist_ok=True)


        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = GaussianPolicy(
        state_dim, action_dim, max_action, dropout=config.actor_dropout
    ).to(config.device)
    true_v_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    semi_v_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    q_network = TwinQ(state_dim, action_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    mu_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    U_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    true_v_optimizer = torch.optim.Adam(true_v_network.parameters(), lr=config.vf_lr)
    semi_v_optimizer = torch.optim.Adam(semi_v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.vf_lr)
    mu_optimizer = torch.optim.Adam(mu_network.parameters(), lr=config.vf_lr)
    U_optimizer = torch.optim.Adam(U_network.parameters(), lr=config.vf_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "true_v_network": true_v_network,
        "true_v_optimizer": true_v_optimizer,
        "semi_v_network": semi_v_network,
        "semi_v_optimizer": semi_v_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "mu_network": mu_network,
        "mu_optimizer": mu_optimizer,
        "U_network": U_network,
        "U_optimizer": U_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # VDICE
        "vdice_type": config.vdice_type,
        "semi_lambda": config.vdice_lambda,
        "semi_eta": config.semi_eta,
        "true_alpha": config.true_alpha,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training VDICE, Env: {config.env_1}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = VDICE(**kwargs)
    return trainer

def change_reward(dataset):
    terminals = dataset["terminals"]
    rewards = dataset["rewards"]

    for i in range(len(terminals)):
        if terminals[i]:
            rewards[i] = 1
        else:
            rewards[i] = 0

    dataset["rewards"] = rewards
    return dataset


def create_dataset(config: TrainConfig):
    # env = gym.make(config.env_1)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]


    # env = gym.make(config.env_1)
    # env_random = gym.make(config.env_2)
    # dataset_e = qlearning_dataset(env)
    # dataset_r = qlearning_dataset(env_random)

    # # combine
    # number = int(config.expert_num)
    # expert_num = number
    # dataset = {
    #     'observations': np.concatenate([dataset_e['observations'][:number], dataset_r['observations'][number:]]),
    #     'actions': np.concatenate([dataset_e['actions'][:number], dataset_r['actions'][number:]]),
    #     'next_observations': np.concatenate([dataset_e['next_observations'][:number], dataset_r['next_observations'][number:]]),
    #     'rewards': np.concatenate([dataset_e['rewards'][:number], dataset_r['rewards'][number:]]),
    #     'terminals': np.concatenate([dataset_e['terminals'][:number], dataset_r['terminals'][number:]]),
    #     'task_horizon': np.concatenate([dataset_e['task_horizon'][:number], dataset_r['task_horizon'][number:]]),
    # }

    # expert_indi = np.array([True] * expert_num)
    # expert_indi = np.append(expert_indi, [False] * (dataset["actions"].shape[0] - expert_num))
    # dataset["expert"] = expert_indi
    # dataset["id"] = np.arange(dataset["actions"].shape[0])

    start_pos = [[15, 3]]
    goal = [[4.5, 15.5]]
    env = PointMassEnv(start=np.array(start_pos[0], dtype=np.float32), action_noise=0)
    dataset = np.load("toy_dataset.npy", allow_pickle=True).item()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset["id"] = np.arange(dataset["actions"].shape[0])

    dataset = change_reward(dataset)

    return dataset, state_dim, action_dim, env

def create_eval_dataset(dataset):

    f_100_exp_dataset = {
        'observations': dataset['observations'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
        'actions': dataset['actions'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
        'next_observations': dataset['next_observations'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
        'rewards': dataset['rewards'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
        'terminals': dataset['terminals'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
        'expert': dataset['expert'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
        'task_horizon': dataset['task_horizon'][np.logical_and(dataset['task_horizon']<100, dataset["expert"])],
    }

    f_100_rand_dataset = {
        'observations': dataset['observations'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
        'actions': dataset['actions'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
        'next_observations': dataset['next_observations'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
        'rewards': dataset['rewards'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
        'terminals': dataset['terminals'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
        'expert': dataset['expert'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
        'task_horizon': dataset['task_horizon'][np.logical_and(dataset['task_horizon']<100, dataset["expert"] == False)],
    }

    l_100_exp_dataset = {
        'observations': dataset['observations'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
        'actions': dataset['actions'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
        'next_observations': dataset['next_observations'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
        'rewards': dataset['rewards'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
        'terminals': dataset['terminals'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
        'expert': dataset['expert'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
        'task_horizon': dataset['task_horizon'][np.logical_and(dataset['task_horizon']>100, dataset["expert"])],
    }

    l_100_rand_dataset = {
        'observations': dataset['observations'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
        'actions': dataset['actions'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
        'next_observations': dataset['next_observations'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
        'rewards': dataset['rewards'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
        'terminals': dataset['terminals'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
        'expert': dataset['expert'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
        'task_horizon': dataset['task_horizon'][np.logical_and(dataset['task_horizon']>100, dataset["expert"] == False)],
    }

    return f_100_exp_dataset, f_100_rand_dataset, l_100_exp_dataset, l_100_rand_dataset

def bc_policy(config, ids, bc_dataset):

    env = gym.make(config.env_1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
        "max_steps": int(1e6),
    }

    # Initialize actor
    trainer = BcLearning(**kwargs)
    preload_filter = np.isin(bc_dataset["id"], ids)
    filtered_dataset = {
                'observations': bc_dataset['observations'][preload_filter],
                'actions': bc_dataset['actions'][preload_filter],
                'next_observations': bc_dataset['next_observations'][preload_filter],
                'rewards': bc_dataset['rewards'][preload_filter],
                'terminals': bc_dataset['terminals'][preload_filter],
                'task_horizon': bc_dataset['task_horizon'][preload_filter],
                'expert': bc_dataset['expert'][preload_filter],
                'id': bc_dataset['id'][preload_filter],
                }

    state_mean, state_std = compute_mean_std(filtered_dataset["observations"], eps=1e-3)

    filtered_dataset["observations"] = normalize_states(
        filtered_dataset["observations"], state_mean, state_std
    )
    filtered_dataset["next_observations"] = normalize_states(
        filtered_dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(filtered_dataset)

    evaluations = []
    for t in range(config.bc_max_timesteps):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        log_dict["bc_step"]= config.bc_policy_step
        wandb.log(log_dict,)
        # Evaluate episode
        # if (t + 1) % config.eval_freq == 0:
        if (t + 1) % 2500 == 0:
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            wandb.log(
                {"bc_train/d4rl_normalized_score": normalized_eval_score,
                 "bc_step": config.bc_policy_step,},
            )
            print("d4rl_normalized_score,",normalized_eval_score)
        config.bc_policy_step += 1
    



@pyrallis.wrap()
def train(config: TrainConfig):
    dataset, state_dim, action_dim, env = create_dataset(config)

    # bc_dataset = copy.deepcopy(dataset)


    # if config.normalize_reward:
    #     modify_reward(dataset, config.env_1)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    # env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)


    config.vdice_lambda = config.semi_vdice_lambda
    semi_trainer = trainer_init(config, env)


    wandb_init(asdict(config))
    # f_100_exp_dataset, f_100_rand_dataset, l_100_exp_dataset, l_100_rand_dataset = create_eval_dataset(dataset)
    evaluations = []
    t = 0

    # import pdb; pdb.set_trace()


    # need_init = True
    # if config.load_semi_model != None and config.load_id_dataset != None:
    #     policy_file = Path(config.load_semi_model)
    #     semi_trainer.load_state_dict(torch.load(policy_file))
    #     t = semi_trainer.total_it
    #     semi_trainer = trainer_init(config, env)
    #     ids = np.load(config.load_id_dataset)
    #     # if id in ids, then keep the data, otherwise remove
    #     preload_filter = np.isin(dataset["id"], ids)
    #     dataset = {
    #         'observations': dataset['observations'][preload_filter],
    #         'actions': dataset['actions'][preload_filter],
    #         'next_observations': dataset['next_observations'][preload_filter],
    #         'rewards': dataset['rewards'][preload_filter],
    #         'terminals': dataset['terminals'][preload_filter],
    #         'task_horizon': dataset['task_horizon'][preload_filter],
    #         'expert': dataset['expert'][preload_filter],
    #         'id': dataset['id'][preload_filter],
    #         }
    #     replay_buffer = ReplayBuffer(
    #         state_dim,
    #         action_dim,
    #         config.buffer_size,
    #         config.device,
    #     )
    #     replay_buffer.load_d4rl_dataset(dataset)
    #     f_100_exp_dataset, f_100_rand_dataset, l_100_exp_dataset, l_100_rand_dataset = create_eval_dataset(dataset)
    #     need_init = False
    #     bc_policy(config, dataset['id'], bc_dataset)



    while t < int(config.max_timesteps):
        
        # if (((t % 1e6) % config.update_freq) == 0 and t > 1e6) or (t % 1e6 == 0 and t > 0 and t < 1e6 + 1) and need_init:
        #     a_weights, s_weights, semi_v = semidice_result_no_vtarget(dataset, semi_trainer, config)

        #     if config.policy_ratio:
        #         preload_filter = a_weights > 0
        #     elif config.state_ratio:
        #         preload_filter = s_weights > 0
        #     elif config.or_ration:
        #         preload_filter = np.logical_or(a_weights > 0, s_weights > 0)
        #     elif config.and_ratio:
        #         preload_filter = np.logical_and(a_weights > 0, s_weights > 0)

        #     dataset = {
        #         'observations': dataset['observations'][preload_filter],
        #         'actions': dataset['actions'][preload_filter],
        #         'next_observations': dataset['next_observations'][preload_filter],
        #         'rewards': dataset['rewards'][preload_filter],
        #         'terminals': dataset['terminals'][preload_filter],
        #         'task_horizon': dataset['task_horizon'][preload_filter],
        #         'expert': dataset['expert'][preload_filter],
        #         'id': dataset['id'][preload_filter],
        #         }
        #     replay_buffer = ReplayBuffer(
        #         state_dim,
        #         action_dim,
        #         config.buffer_size,
        #         config.device,
        #     )
        #     replay_buffer.load_d4rl_dataset(dataset)
        #     f_100_exp_dataset, f_100_rand_dataset, l_100_exp_dataset, l_100_rand_dataset = create_eval_dataset(dataset)
        #     config.vdice_type = "semi"
        #     config.vdice_lambda = semi_trainer.vdice_lambda + config.semi_lambda_delta
        #     # config.semi_lambda_delta = config.semi_lambda_delta * 0.25
        #     semi_trainer = trainer_init(config, env)

        #     bc_policy(config, dataset['id'], bc_dataset)

        # need_init = True

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        semi_log_dict = semi_trainer.train(batch)
        semi_log_dict["vdice_step"] = t

        

        wandb.log(semi_log_dict,)
        # print("t: ", t)

        # Evaluate episode
        # if (t + 1) % config.eval_freq == 0:
            
        #     a_weights_exp, s_weights_exp, semi_v_exp = semidice_result_no_vtarget(f_100_exp_dataset, semi_trainer, config)
        #     a_weights_rand, s_weights_rand, semi_v_rand = semidice_result_no_vtarget(f_100_rand_dataset, semi_trainer, config)
        #     a_weights_l_100_exp, s_weights_l_100_exp, semi_l_100_v_exp = semidice_result_no_vtarget(l_100_exp_dataset, semi_trainer, config)
        #     a_weights_l_100_rand, s_weights_l_100_rand, semi_l_100_v_rand = semidice_result_no_vtarget(l_100_rand_dataset, semi_trainer, config)
        #     a_weights, s_weights, semi_v = semidice_result_no_vtarget(dataset, semi_trainer, config)

        #     if config.policy_ratio:
        #         preload_filter = a_weights > 0
        #     elif config.state_ratio:
        #         preload_filter = s_weights > 0
        #     elif config.or_ration:
        #         preload_filter = np.logical_or(a_weights > 0, s_weights > 0)
        #     elif config.and_ratio:
        #         preload_filter = np.logical_and(a_weights > 0, s_weights > 0)

        #     filter_size = preload_filter.sum()
        #     expert_in_filter = dataset["expert"][np.logical_and(preload_filter, dataset["expert"])].sum()
        #     expert_in_filter_f100 = dataset["expert"][np.logical_and(dataset["task_horizon"] < 100, np.logical_and(preload_filter, dataset["expert"]))].sum()
        #     random_in_filter = filter_size - expert_in_filter

        #     expert_in_filter_f25 = dataset["expert"][np.logical_and(dataset["task_horizon"] < 25, np.logical_and(preload_filter, dataset["expert"]))].sum()
        #     random_in_filter_f25 = np.logical_and(dataset["task_horizon"] < 25, np.logical_and(preload_filter, dataset["expert"] == False)).sum()

        #     print("==========================================================")
        #     print("==========================================================")
        #     print("Semi F100 Expert Action Weights: ", a_weights_exp.mean())
        #     print("Semi F100 Expert Action Weights Std: ", a_weights_exp.std())
        #     print("Semi F100 Expert State Weights: ", s_weights_exp.mean())
        #     print("Semi F100 Expert State Weights Std: ", s_weights_exp.std())
        #     print("Semi F100 Expert V: ", semi_v_exp.mean())

        #     print("Semi F100 Random Action Weights: ", a_weights_rand.mean())
        #     print("Semi F100 Random Action Weights Std: ", a_weights_rand.std())
        #     print("Semi F100 Random State Weights: ", s_weights_rand.mean())
        #     print("Semi F100 Random State Weights Std: ", s_weights_rand.std())
        #     print("Semi F100 Random V: ", semi_v_rand.mean())


        #     print("Semi L100 Expert Action Weights: ", a_weights_l_100_exp.mean())
        #     print("Semi L100 Expert Action Weights Std: ", a_weights_l_100_exp.std())
        #     print("Semi L100 Expert State Weights: ", s_weights_l_100_exp.mean())
        #     print("Semi L100 Expert State Weights Std: ", s_weights_l_100_exp.std())
        #     print("Semi L100 Expert V: ", semi_l_100_v_exp.mean())


        #     print("Semi L100 Random Action Weights: ", a_weights_l_100_rand.mean())
        #     print("Semi L100 Random Action Weights Std: ", a_weights_l_100_rand.std())
        #     print("Semi L100 Random State Weights: ", s_weights_l_100_rand.mean())
        #     print("Semi L100 Random State Weights Std: ", s_weights_l_100_rand.std())
        #     print("Semi L100 Random V: ", semi_l_100_v_rand.mean())


        #     print("Semi Dataset Action Weights: ", a_weights.mean())
        #     print("Semi Dataset Action Weights Std: ", a_weights.std())
        #     print("Semi Dataset State Weights: ", s_weights.mean())
        #     print("Semi Dataset State Weights Std: ", s_weights.std())
        #     print("Semi Dataset V: ", semi_v.mean())


        #     print("Num Filter Size: ", filter_size)
        #     print("Num Expert in Filter: ", expert_in_filter)
        #     print("Num F100 Expert in Filter: ", expert_in_filter_f100)
        #     print("Num Random in Filter: ", random_in_filter)
        #     print("Num F25 Expert in Filter: ", expert_in_filter_f25)
        #     print("Num F25 Random in Filter: ", random_in_filter_f25)



        #     wandb.log(
        #         {
        #             "vdice_train/Semi F100 Expert Action Weights: ": a_weights_exp.mean(),
        #             "vdice_train/Semi F100 Expert Action Weights Std: ": a_weights_exp.std(),
        #             "vdice_train/Semi F100 Expert State Weights: ": s_weights_exp.mean(),
        #             "vdice_train/Semi F100 Expert State Weights Std: ": s_weights_exp.std(),
        #             "vdice_train/Semi F100 Expert V: ": semi_v_exp.mean(),

        #             "vdice_train/Semi F100 Random Action Weights: ": a_weights_rand.mean(),
        #             "vdice_train/Semi F100 Random Action Weights Std: ": a_weights_rand.std(),
        #             "vdice_train/Semi F100 Random State Weights: ": s_weights_rand.mean(),
        #             "vdice_train/Semi F100 Random State Weights Std: ": s_weights_rand.std(),
        #             "vdice_train/Semi F100 Random V: ": semi_v_rand.mean(),


        #             "vdice_train/Semi L100 Expert Action Weights: ": a_weights_l_100_exp.mean(),
        #             "vdice_train/Semi L100 Expert Action Weights Std: ": a_weights_l_100_exp.std(),
        #             "vdice_train/Semi L100 Expert State Weights: ": s_weights_l_100_exp.mean(),
        #             "vdice_train/Semi L100 Expert State Weights Std: ": s_weights_l_100_exp.std(),
        #             "vdice_train/Semi L100 Expert V: ": semi_l_100_v_exp.mean(),


        #             "vdice_train/Semi L100 Random Action Weights: ": a_weights_l_100_rand.mean(),
        #             "vdice_train/Semi L100 Random Action Weights Std: ": a_weights_l_100_rand.std(),
        #             "vdice_train/Semi L100 Random State Weights: ": s_weights_l_100_rand.mean(),
        #             "vdice_train/Semi L100 Random State Weights Std: ": s_weights_l_100_rand.std(),
        #             "vdice_train/Semi L100 Random V: ": semi_l_100_v_rand.mean(),


        #             "vdice_train/Semi Dataset Action Weights: ": a_weights.mean(),
        #             "vdice_train/Semi Dataset Action Weights Std: ": a_weights.std(),
        #             "vdice_train/Semi Dataset State Weights: ": s_weights.mean(),
        #             "vdice_train/Semi Dataset State Weights Std: ": s_weights.std(),
        #             "vdice_train/Semi Dataset V: ": semi_v.mean(),
                    

        #             "vdice_train/Num Filter Size": filter_size,
        #             "vdice_train/Num Expert in Filter": expert_in_filter,
        #             "vdice_train/Num F100 Expert in Filter": expert_in_filter_f100,
        #             "vdice_train/Num Random in Filter": random_in_filter,
        #             "vdice_train/Semi Lambda": config.vdice_lambda,
        #             "vdice_train/Num F25 Expert in Filter": expert_in_filter_f25,
        #             "vdice_train/Num F25 Random in Filter": random_in_filter_f25,
        #             "vdice_step": t,
        #             },
        #     )

            

            


        if (t + 1) % config.save_freq == 0:
            if config.checkpoints_path is not None:
                torch.save(
                    semi_trainer.state_dict(),
                    os.path.join(config.checkpoints_path+"_semi", f"checkpoint_{t}.pt"),
                )

                a_weights, s_weights, semi_v = semidice_result_no_vtarget(dataset, semi_trainer, config)
                action_ratio_filter = a_weights > 0
                state_ratio_filter = s_weights > 0
                or_ratio_filter = np.logical_or(a_weights > 0, s_weights > 0)
                and_ratio_filter = np.logical_and(a_weights > 0, s_weights > 0)

                log_dict = {}
                log_dict["filter/size of action ratio filter: "] = action_ratio_filter.sum()
                log_dict["filter/size of state ratio filter: "] = state_ratio_filter.sum()
                log_dict["filter/size of or ratio filter: "] = or_ratio_filter.sum()
                log_dict["filter/size of and ratio filter: "] = and_ratio_filter.sum()
                wandb.log(log_dict)

                # save dataset id
                np.save(os.path.join(config.checkpoints_path+"_action_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][action_ratio_filter])
                np.save(os.path.join(config.checkpoints_path+"_state_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][state_ratio_filter])
                np.save(os.path.join(config.checkpoints_path+"_or_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][or_ratio_filter])
                np.save(os.path.join(config.checkpoints_path+"_and_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][and_ratio_filter])
        t += 1


if __name__ == "__main__":
    train()