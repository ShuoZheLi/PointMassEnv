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
from utils import TwinQ, ValueFunction, GaussianPolicy, DeterministicPolicy, \
                    ReplayBuffer, MLP, TwinV, soft_update, set_seed, compute_mean_std, \
                    eval_actor, return_reward_range, modify_reward, normalize_states, \
                    wrap_env, BcLearning, qlearning_dataset

from PointMassEnv import PointMassEnv, WALLS
import imageio


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
        
        self.true_v_optimizer = true_v_optimizer
        self.semi_v_optimizer = semi_v_optimizer
        self.q_optimizer = q_optimizer
        self.mu_optimizer = mu_optimizer
        self.U_optimizer = U_optimizer

        self.semi_sa_actor_and = actor
        self.semi_sa_actor_and_optimizer = actor_optimizer
        self.semi_sa_actor_and_lr_schedule = CosineAnnealingLR(self.semi_sa_actor_and_optimizer, max_steps)

        self.semi_sa_actor_or = copy.deepcopy(self.semi_sa_actor_and).to(device)
        self.semi_sa_actor_or_optimizer = torch.optim.Adam(self.semi_sa_actor_or.parameters(), lr=3e-4)
        self.semi_sa_actor_or_lr_schedule = CosineAnnealingLR(self.semi_sa_actor_or_optimizer, max_steps)

        self.semi_s_actor = copy.deepcopy(self.semi_sa_actor_and).to(device)
        self.semi_s_actor_optimizer = torch.optim.Adam(self.semi_s_actor.parameters(), lr=3e-4)
        self.semi_s_actor_lr_schedule = CosineAnnealingLR(self.semi_s_actor_optimizer, max_steps)

        self.semi_a_actor = copy.deepcopy(self.semi_sa_actor_and).to(device)
        self.semi_a_actor_optimizer = torch.optim.Adam(self.semi_a_actor.parameters(), lr=3e-4)
        self.semi_a_actor_lr_schedule = CosineAnnealingLR(self.semi_a_actor_optimizer, max_steps)

        self.true_sa_actor = copy.deepcopy(self.semi_sa_actor_and).to(device)
        self.true_sa_actor_optimizer = torch.optim.Adam(self.true_sa_actor.parameters(), lr=3e-4)
        self.true_sa_actor_lr_schedule = CosineAnnealingLR(self.true_sa_actor_optimizer, max_steps)

        self.vdice_type = vdice_type
        self.vdice_lambda = semi_lambda
        self.semi_eta = semi_eta
        self.true_alpha = true_alpha
        self.f_name = f_name
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def semidice_result_no_vtarget(self, dataset):

        a_weights = []
        s_weights = []
        semi_vs = []
        for i in range(0,len(dataset["observations"]),8192):

                # v_s.append(v.mean(0).cpu().numpy())

            with torch.no_grad():
                observations = torch.FloatTensor(dataset["observations"][i:min(i+8192,len(dataset["observations"]))]).to(self.device)
                actions = torch.FloatTensor(dataset["actions"][i:min(i+8192,len(dataset["actions"]))]).to(self.device)
                
                target_q = self.q_target(observations, actions)
                semi_v = self.semi_v(observations)
                adv = (target_q - semi_v)
                a_weight = f_prime_inverse(self.f_name, adv)

                u = self.U(observations)
                s_weight = f_prime_inverse(self.f_name, u)

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
        # TODO: why frenchel_duaL here
        forward_dual_loss = self.vdice_lambda * frenchel_dual(self.f_name, adv)
        semi_v_loss = semi_linear_loss + forward_dual_loss
        semi_v_loss = torch.mean(semi_v_loss)
        self.semi_v_optimizer.zero_grad()
        semi_v_loss.backward()
        self.semi_v_optimizer.step()

        semi_pi_residual = adv.clone().detach()

        true_v = self.true_v(observations)
        true_v_next = self.true_v(next_observations)
        true_residual = rewards + (1.0 - terminals.float()) * self.discount * true_v_next - true_v
        true_residual = true_residual / self.true_alpha
        true_dual_loss = torch.mean(frenchel_dual(self.f_name, true_residual))
        # TODO: why is there a discount factor here?
        # shouldn't it be the lambda ??
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
        # u = self.U(observations)
        # s_weight = f_prime_inverse(self.f_name, u)
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
        # TODO: where is the reward??
        mu_residual = (1.0 - terminals.float()) * self.discount * mu_next - mu
        
        # !!!!!!!!!!!!!!!! why f_prime_inverse, not frenchel_dual_prime
        a_weight = f_prime_inverse(self.f_name, semi_pi_residual)
        u_target = a_weight * mu_residual
        # TODO: self.U is learned with weighted loss, its value should be weighted already,
        # so we don't need to a_weight it again
        # TODO: why frenchel_dual_prime here not frenchel_dual??
        s_a_weight = (a_weight * f_prime_inverse(self.f_name, u)).clone().detach()

        u_loss = F.mse_loss(u, u_target)

        self.U_optimizer.zero_grad()
        u_loss.backward()
        self.U_optimizer.step()

        log_dict["vdice_train/a_weight"] = a_weight.mean().item()
        log_dict["vdice_train/U_loss"] = u_loss.item()
        log_dict["vdice_train/U_value"] = u.mean().item()

        return s_a_weight

    def _update_mu(
        self,
        s_a_weight: torch.Tensor,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
        init_observations: torch.Tensor,
        log_dict: Dict,
    ):
        mu = self.mu(observations)
        mu_next = self.mu(next_observations)
        # TODO: where is the reward??
        mu_residual = (1.0 - terminals.float()) * self.discount * mu_next - mu
        # TODO: why frenchel_dual?? !!!!!!!!!!!!!!!!!!!!!!!!!!! frenchel_dual NOOOOOOOOO
        mu_dual_loss = torch.mean(s_a_weight * mu_residual)
        # TODO: why is there a discount factor here?
        mu_linear_loss = (1 - self.discount) * torch.mean(self.mu(init_observations))
        mu_loss = mu_linear_loss + mu_dual_loss

        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        log_dict["vdice_train/mu_loss"] = mu_loss.item()
        log_dict["vdice_train/mu_value"] = mu.mean().item()


    def _update_policy(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            semi_v = self.semi_v(observations)
            adv = target_q - semi_v
            semi_a_weight = f_prime_inverse(self.f_name, adv)
            u = self.U(observations)
            semi_s_weight = f_prime_inverse(self.f_name, u)

            true_v = self.true_v(observations)
            true_v_next = self.true_v(next_observations)
            true_residual = rewards + (1.0 - terminals.float()) * self.discount * true_v_next - true_v
            true_residual = true_residual / self.true_alpha
            true_sa_weight = f_prime_inverse(self.f_name, true_residual)





        policy_out = self.semi_sa_actor_and(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        policy_loss = torch.mean(semi_a_weight * semi_s_weight * bc_losses)
        self.semi_sa_actor_and_optimizer.zero_grad()
        policy_loss.backward()
        self.semi_sa_actor_and_optimizer.step()
        self.semi_sa_actor_and_lr_schedule.step()
        log_dict["vdice_train/semi_sa_policy_loss"] = policy_loss.item()

        policy_out = self.semi_sa_actor_or(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        semi_a_or_s_weight = torch.max(semi_a_weight, semi_s_weight)
        policy_loss = torch.mean(semi_a_or_s_weight * bc_losses)
        self.semi_sa_actor_or_optimizer.zero_grad()
        policy_loss.backward()
        self.semi_sa_actor_or_optimizer.step()
        self.semi_sa_actor_or_lr_schedule.step()
        log_dict["vdice_train/semi_sa_or_policy_loss"] = policy_loss.item()

        policy_out = self.semi_s_actor(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        policy_loss = torch.mean(semi_s_weight * bc_losses)
        self.semi_s_actor_optimizer.zero_grad()
        policy_loss.backward()
        self.semi_s_actor_optimizer.step()
        self.semi_s_actor_lr_schedule.step()
        log_dict["vdice_train/semi_s_policy_loss"] = policy_loss.item()

        policy_out = self.semi_a_actor(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        policy_loss = torch.mean(semi_a_weight * bc_losses)
        self.semi_a_actor_optimizer.zero_grad()
        policy_loss.backward()
        self.semi_a_actor_optimizer.step()
        self.semi_a_actor_lr_schedule.step()
        log_dict["vdice_train/semi_a_policy_loss"] = policy_loss.item()

        policy_out = self.true_sa_actor(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        policy_loss = torch.mean(true_sa_weight * bc_losses)
        self.true_sa_actor_optimizer.zero_grad()
        policy_loss.backward()
        self.true_sa_actor_optimizer.step()
        self.true_sa_actor_lr_schedule.step()
        log_dict["vdice_train/true_sa_policy_loss"] = policy_loss.item()
        

    
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
        log_dict = {}
        flags = torch.ones_like(rewards)

        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update V function
        semi_residual, true_residual = self._update_v(observations, 
                                                      actions, 
                                                      next_observations, 
                                                      rewards, 
                                                      dones, 
                                                      observations, 
                                                      log_dict)
        # Update Q function
        self._update_q(observations, 
                       actions, 
                       next_observations, 
                       rewards, 
                       dones, 
                       log_dict)
        
        # Update U function
        s_a_weight = self._update_U(semi_residual, 
                                    observations, 
                                    next_observations, 
                                    dones, 
                                    log_dict)

        # Update Mu function
        self._update_mu(s_a_weight, 
                        observations, 
                        next_observations, 
                        dones, 
                        observations, 
                        log_dict)

        # Update actor
        self._update_policy(observations, 
                            next_observations, 
                            actions, 
                            rewards, 
                            dones, 
                            log_dict,)

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

            "semi_sa_actor_and": self.semi_sa_actor_and.state_dict(),
            "semi_sa_actor_and_optimizer": self.semi_sa_actor_and_optimizer.state_dict(),
            "semi_sa_actor_or": self.semi_sa_actor_or.state_dict(),
            "semi_sa_actor_or_optimizer": self.semi_sa_actor_or_optimizer.state_dict(),
            "semi_s_actor": self.semi_s_actor.state_dict(),
            "semi_s_actor_optimizer": self.semi_s_actor_optimizer.state_dict(),
            "semi_a_actor": self.semi_a_actor.state_dict(),
            "semi_a_actor_optimizer": self.semi_a_actor_optimizer.state_dict(),
            "true_sa_actor": self.true_sa_actor.state_dict(),
            "true_sa_actor_optimizer": self.true_sa_actor_optimizer.state_dict(),

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
        self.semi_a_actor_and.load_state_dict(state_dict["semi_sa_actor_and"])
        self.semi_a_actor_and_optimizer.load_state_dict(state_dict["semi_sa_actor_and_optimizer"])
        self.semi_a_actor_or.load_state_dict(state_dict["semi_sa_actor_or"])
        self.semi_a_actor_or_optimizer.load_state_dict(state_dict["semi_sa_actor_or_optimizer"])
        self.semi_s_actor.load_state_dict(state_dict["semi_s_actor"])
        self.semi_s_actor_optimizer.load_state_dict(state_dict["semi_s_actor_optimizer"])
        self.semi_a_actor.load_state_dict(state_dict["semi_a_actor"])
        self.semi_a_actor_optimizer.load_state_dict(state_dict["semi_a_actor_optimizer"])
        self.true_sa_actor.load_state_dict(state_dict["true_sa_actor"])
        self.true_sa_actor_optimizer.load_state_dict(state_dict["true_sa_actor_optimizer"])
        self.total_it = state_dict["total_it"]



def trainer_init(config: TrainConfig, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        # os.makedirs(config.checkpoints_path+"/semi_s_and_a/model", exist_ok=True)
        # os.makedirs(config.checkpoints_path+"/semi_s_or_a/model", exist_ok=True)
        # os.makedirs(config.checkpoints_path+"/semi_s/model", exist_ok=True)
        # os.makedirs(config.checkpoints_path+"/semi_a/model", exist_ok=True)
        # os.makedirs(config.checkpoints_path+"/true_s_and_a/model", exist_ok=True)

        os.makedirs(config.checkpoints_path+"/model", exist_ok=True)

        os.makedirs(config.checkpoints_path+"/semi_s_and_a/gif", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/semi_s_or_a/gif", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/semi_s/gif", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/semi_a/gif", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/true_s_and_a/gif", exist_ok=True)


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

    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                                goal=np.array([4.5, 12.5], dtype=np.float32), 
                                goal_radius=0.8)
    dataset = np.load("dataset.npy", allow_pickle=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset["id"] = np.arange(dataset["actions"].shape[0])

    # dataset = change_reward(dataset)

    return dataset, state_dim, action_dim, env

def eval_policy(actor, global_step, gif_dir, device, wandb, name):
    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8)
    
    actor.eval()
    images = []
    count_success = 0
    for i in range(5):
        episode_return = 0.0
        episode_length = 0
        done = False
        obs, _ = env.reset()
        images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
        while not done:
            with torch.no_grad():
                mean = actor.act(torch.Tensor([obs]).to(device), device=device)
            obs, reward, done, trunc, info = env.step(mean)
            images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
            episode_return += reward
            episode_length += 1
            if done and info["success"]:
                count_success += 1
    
    actor.train()
    # save images into gif
    imageio.mimsave(gif_dir + "/" +str(global_step) + ".gif", images, fps=10)
    success_rate = count_success / 5.0

    wandb.log(
                {"policy_train/"+name: success_rate,
                 "policy_train": global_step,},
            )
    print(name+f" Success rate: {success_rate}")
    return episode_return, episode_length


@pyrallis.wrap()
def train(config: TrainConfig):
    dataset, state_dim, action_dim, env = create_dataset(config)

    if config.normalize_reward:
        modify_reward(dataset, config.env_1)

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
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

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
    evaluations = []
    t = 0


    while t < int(config.max_timesteps):
        

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        semi_log_dict = semi_trainer.train(batch)
        semi_log_dict["vdice_step"] = t        

        wandb.log(semi_log_dict,)

        if (t + 1) % config.eval_freq == 0:
            eval_policy(semi_trainer.semi_sa_actor_and, t, config.checkpoints_path+"/semi_s_and_a/gif", config.device, wandb, "semi_s_and_a")
            eval_policy(semi_trainer.semi_sa_actor_or, t, config.checkpoints_path+"/semi_s_or_a/gif", config.device, wandb, "semi_s_or_a")
            eval_policy(semi_trainer.semi_s_actor, t, config.checkpoints_path+"/semi_s/gif", config.device, wandb, "semi_s")
            eval_policy(semi_trainer.semi_a_actor, t, config.checkpoints_path+"/semi_a/gif", config.device, wandb, "semi_a")
            eval_policy(semi_trainer.true_sa_actor, t, config.checkpoints_path+"/true_s_and_a/gif", config.device, wandb, "true_s_and_a")
            print("==============================")

        if (t + 1) % config.save_freq == 0:
            if config.checkpoints_path is not None:
                torch.save(
                    semi_trainer.state_dict(),
                    os.path.join(config.checkpoints_path+"/model", f"checkpoint_{t}.pt"),
                )

                # a_weights, s_weights, semi_v = semidice_result_no_vtarget(dataset, semi_trainer, config)
                # action_ratio_filter = a_weights > 0
                # state_ratio_filter = s_weights > 0
                # or_ratio_filter = np.logical_or(a_weights > 0, s_weights > 0)
                # and_ratio_filter = np.logical_and(a_weights > 0, s_weights > 0)


                # # save dataset id
                # np.save(os.path.join(config.checkpoints_path+"_action_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][action_ratio_filter])
                # np.save(os.path.join(config.checkpoints_path+"_state_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][state_ratio_filter])
                # np.save(os.path.join(config.checkpoints_path+"_or_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][or_ratio_filter])
                # np.save(os.path.join(config.checkpoints_path+"_and_ratio_id", f"checkpoint_{t}.npy"), dataset["id"][and_ratio_filter])
        t += 1


if __name__ == "__main__":
    train()