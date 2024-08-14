# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os, sys
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import d4rl
import gymnasium as gym
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offline_RL.utils import TwinQ, ValueFunction, GaussianPolicy, DeterministicPolicy, \
                    ReplayBuffer, MLP, TwinV, soft_update, set_seed, compute_mean_std, \
                    eval_actor, return_reward_range, modify_reward, normalize_states, \
                    wrap_env, BcLearning, qlearning_dataset, SingleQ

from PointMassEnv import PointMassEnv, WALLS
import imageio
import yaml

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TrainConfig:
    #############################
    ######### Experiment ########
    #############################
    seed: int = 100
    eval_freq: int = int(10)  # How often (time steps) we evaluate
    save_freq: int = int(10)  # How often (time steps) save the model
    update_freq: int = int(1.5e5)  # How often (time steps) we update the model
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    # discount: float = 0.99  # Discount factor
    discount: float = 0.976  # Discount factor

    #############################
    ######### NN Arc ############
    #############################
    vf_lr: float = 3e-4  # V function learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    layernorm: bool = False
    hidden_dim: int = 256
    tau: float = 0.005  # Target network update rate
    
    #############################
    ###### dataset preprocess ###
    #############################
    # normalize: bool = True  # Normalize states
    normalize: bool = False  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    # normalize_reward: bool = False  # Normalize reward
    
    #############################
    ###### Wandb Logging ########
    #############################
    project: str = "test"
    checkpoints_path: Optional[str] = "test"
    load_model: str = ""  # Model load file name, "" doesn't load
    load_yaml: str = ""  # Model load file name, "" doesn't load
    alg: str = "test"
    env_1: str = "antmaze-umaze-v2"  # OpenAI gym environment name
    env_2: str = "antmaze-umaze-v2"  # OpenAI gym environment name

    #############################
    #### DICE Hyperparameters ###
    #############################
    device: str = "cuda"
    vdice_type: Optional[str] = "semi"
    semi_dice_lambda: float = 0.3
    true_dice_alpha: float = 1.0
    env_name: str = "FourRooms"
    reward_type: str = "sparse"
    percent_expert: float = 0
    discrete_action: bool = False

    # semi_q_reward: bool = True
    semi_q_alpha: float = 1.0
    

    # def __post_init__(self):
    #     self.name = f"{self.alg}-{self.env_1}-{str(uuid.uuid4())[:8]}"
    #     if self.checkpoints_path is not None:
    #         self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)



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
    wandb.save("*.py")

    wandb.define_metric("vdice_step")
    wandb.define_metric("value_step")
    wandb.define_metric("selected_traj_step")
    wandb.define_metric("selected_expert_traj_step")
    wandb.define_metric("policy_step")


    wandb.define_metric("vdice_loss/*", step_metric="vdice_step")
    wandb.define_metric("value_train/*", step_metric="value_step")
    wandb.define_metric("selected_traj/*", step_metric="selected_traj_step")
    wandb.define_metric("selected_expert_traj/*", step_metric="selected_expert_traj_step")
    wandb.define_metric("policy_train/*", step_metric="policy_step")


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
    
# def frenchel_dual_q(name, x):
#     return torch.max(x + x**2 / 4, torch.zeros_like(x))

# def frenchel_dual_q(name, x):
#     x = torch.abs(x)
#     return 1/3 * x**3

# def frenchel_dual_q_prime_inverse(name, x):
#     return torch.sign(x) * torch.sqrt(torch.abs(x))

def frenchel_dual_q(name, x):
    return x**2

def frenchel_dual_q_prime_inverse(name, x):
    return x


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

        semi_q: nn.Module,
        semi_q_optimizer: torch.optim.Optimizer,

        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        mu_network: nn.Module,
        mu_optimizer: torch.optim.Optimizer,
        U_network: nn.Module,
        U_optimizer: torch.optim.Optimizer,
        vdice_type: str = "semi",
        semi_dice_lambda: float = 0.7,
        true_dice_alpha: float = 0.1,
        f_name: str = 'Smoothed_square_chi',
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        semi_q_alpha: float = 1,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.true_v = true_v_network
        self.semi_v = semi_v_network
        self.q = q_network
        self.q_target = copy.deepcopy(self.q).requires_grad_(False).to(device)
        self.mu = mu_network
        self.U = U_network

        # TwinQ(state_dim, action_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
        # SingleQ(state_dim, action_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)

        self.semi_q = semi_q
        self.semi_q_optimizer = semi_q_optimizer
        
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

        self.bc_actor = copy.deepcopy(self.semi_sa_actor_and).to(device)
        self.bc_actor_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=3e-4)
        self.bc_actor_lr_schedule = CosineAnnealingLR(self.bc_actor_optimizer, max_steps)

        self.vdice_type = vdice_type
        self.semi_dice_lambda = semi_dice_lambda
        self.true_dice_alpha = true_dice_alpha
        self.f_name = f_name
        self.discount = discount
        self.tau = tau

        self.semi_q_alpha = semi_q_alpha

        self.total_it = int(0)
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

        # semi_v_linear = self.semi_v(init_observations)
        semi_v = semi_v_linear = self.semi_v(observations)
        adv = target_q - semi_v
        semi_linear_loss = (1 - self.semi_dice_lambda) * semi_v_linear
        # TODO: why frenchel_duaL here
        forward_dual_loss = self.semi_dice_lambda * frenchel_dual(self.f_name, adv)
        semi_v_loss = semi_linear_loss + forward_dual_loss
        semi_v_loss = torch.mean(semi_v_loss)
        self.semi_v_optimizer.zero_grad()
        semi_v_loss.backward()
        self.semi_v_optimizer.step()

        semi_pi_residual = adv.clone().detach()

        true_v = self.true_v(observations)
        true_v_next = self.true_v(next_observations)
        true_residual = rewards + (1.0 - terminals.float()) * self.discount * true_v_next - true_v
        true_residual = true_residual / self.true_dice_alpha
        true_dual_loss = torch.mean(frenchel_dual(self.f_name, true_residual))
        # TODO: why is there a discount factor here?
        # shouldn't it be the lambda ??
        true_linear_loss = (1 - self.discount) * torch.mean(self.true_v(init_observations))
        true_v_loss = true_linear_loss + true_dual_loss
        true_pi_residual = true_residual.clone().detach()

        self.true_v_optimizer.zero_grad()
        true_v_loss.backward()
        self.true_v_optimizer.step()

        log_dict["value_train/semi_v_value"] = semi_v.mean().item()
        log_dict["value_train/true_v_value"] = true_v.mean().item()

        # Update target Q network
        # soft_update(self.v_target, self.v, self.tau)
        return semi_pi_residual, true_pi_residual
    
    # def _update_semi_q(
    #     self,
    #     observations: torch.Tensor,
    #     actions: torch.Tensor,
    #     next_observations: torch.Tensor,
    #     rewards: torch.Tensor,
    #     terminals: torch.Tensor,
    #     init_observations: torch.Tensor,
    #     log_dict: Dict,
    # ):
    #     with torch.no_grad():
    #         init_a = self.semi_a_actor(init_observations).mean
    #         next_a = self.semi_a_actor(next_observations).mean
        
    #     next_q = self.semi_q(next_observations, next_a)
    #     targets = (1.0 - terminals.float()) * self.discount * next_q
    #     # targets = rewards + (1.0 - terminals.float()) * self.discount * next_q

    #     semi_q = self.semi_q(observations, actions)
    #     adv = targets - semi_q
    #     # forward_dual_loss = torch.mean(self.semi_q_alpha * frenchel_dual_q(self.f_name, adv / self.semi_q_alpha))
    #     forward_dual_loss = torch.mean(self.semi_q_alpha * frenchel_dual(self.f_name, adv / self.semi_q_alpha))

    #     semi_q_linear = self.semi_q(init_observations, init_a)
    #     semi_q_linear_loss = torch.mean((1 - self.discount) *  semi_q_linear)
        
    #     semi_q_loss = semi_q_linear_loss + forward_dual_loss
        
    #     log_dict["value_train/semi_q_value"] = semi_q.mean().item()
    #     log_dict["vdice_loss/semi_q_loss"] = semi_q_loss.item()
    #     self.semi_q_optimizer.zero_grad()
    #     semi_q_loss.backward()
    #     self.semi_q_optimizer.step()

    
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
        log_dict["vdice_loss/q_loss"] = q_loss.item()
        log_dict["value_train/q_value"] = qs[0].mean().item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.q, self.tau)

    # def _update_U(
    #     self,
    #     semi_pi_residual: torch.Tensor,
    #     observations: torch.Tensor,
    #     next_observations: torch.Tensor,
    #     terminals: torch.Tensor,
    #     log_dict: Dict,
    # ):
    #     u = self.U(observations)
    #     mu = self.mu(observations)
    #     mu_next = self.mu(next_observations)
    #     # TODO: where is the reward??
    #     mu_residual = (1.0 - terminals.float()) * self.discount * mu_next - mu
        
    #     # !!!!!!!!!!!!!!!! why f_prime_inverse, not frenchel_dual_prime
    #     a_weight = f_prime_inverse(self.f_name, semi_pi_residual)
    #     u_target = a_weight * mu_residual
    #     # TODO: self.U is learned with weighted loss, its value should be weighted already,
    #     # so we don't need to a_weight it again
    #     # TODO: why frenchel_dual_prime here not frenchel_dual??
    #     s_a_weight = (a_weight * f_prime_inverse(self.f_name, u)).clone().detach()

    #     u_loss = F.mse_loss(u, u_target)

    #     self.U_optimizer.zero_grad()
    #     u_loss.backward()
    #     self.U_optimizer.step()

    #     log_dict["vdice_loss/U_loss"] = u_loss.item()
    #     log_dict["value_train/U_value"] = u.mean().item()

    #     return s_a_weight

    def _update_mu(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
        init_observations: torch.Tensor,
        actions_list: list,
        next_states_list: list,
        dones_list: list,
        log_dict: Dict,
    ):
        
        # semi_a_weighted_residual_list = []
        
        # for i in range(observations.shape[0]):
        #     with torch.no_grad():
        #         target_q = self.q_target(observations[i].repeat(actions_list[i].shape[0], 1), actions_list[i])
        #         semi_v = self.semi_v(observations[i].repeat(actions_list[i].shape[0], 1))
        #         adv = target_q - semi_v
        #         semi_a_weight = f_prime_inverse(self.f_name, adv)
            
            

        #     mu = self.mu(observations[i].repeat(next_states_list[i].shape[0], 1))
        #     mu_next = self.mu(next_states_list[i])
        #     mu_residual = (1.0 - torch.squeeze(dones_list[i], dim=1)) * self.discount * mu_next - mu

        #     semi_a_weighted_residual_list.append(mu_residual * semi_a_weight)
        #     semi_a_weighted_residual_list[-1] = semi_a_weighted_residual_list[-1].mean()
        #     semi_a_weighted_residual_list[-1] = self.semi_q_alpha * frenchel_dual(self.f_name, semi_a_weighted_residual_list[-1] / self.semi_q_alpha)

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # import pdb; pdb.set_trace()
        # Calculate target_q and semi_v for all observations and actions in a vectorized manner
        with torch.no_grad():
            repeated_observations = torch.cat([obs.repeat(act.shape[0], 1) for obs, act in zip(observations, actions_list)], dim=0)
            repeated_actions = torch.cat(actions_list, dim=0)
            
            target_q = self.q_target(repeated_observations, repeated_actions)
            semi_v = self.semi_v(repeated_observations)
            
            # Calculate advantages and semi_a_weights for all
            adv = target_q - semi_v
            semi_a_weight = f_prime_inverse(self.f_name, adv)

        # Calculate mu and mu_residual in a vectorized manner
        repeated_obs_for_mu = torch.cat([obs.repeat(next_states.shape[0], 1) for obs, next_states in zip(observations, next_states_list)], dim=0)
        repeated_next_states = torch.cat(next_states_list, dim=0)

        mu = self.mu(repeated_obs_for_mu)
        mu_next = self.mu(repeated_next_states)

        dones_flat = torch.cat(dones_list, dim=0)
        mu_residual = (1.0 - torch.squeeze(dones_flat, dim=1)) * self.discount * mu_next - mu

        # Apply semi_a_weight to mu_residual
        semi_a_weighted_residual = mu_residual * semi_a_weight

        # Create an index map to group residuals by observations
        index_map = torch.cat([torch.full((act.shape[0],), i, dtype=torch.long) for i, act in enumerate(actions_list)]).to(self.device)

        # Use scatter_add to sum up residuals per observation and divide by count to get the mean
        semi_a_weighted_residual_sum = torch.zeros(observations.shape[0], device=self.device).scatter_add_(0, index_map, semi_a_weighted_residual)
        action_counts = torch.tensor([act.shape[0] for act in actions_list], dtype=torch.float32, device=self.device)
        semi_a_weighted_residual_mean = semi_a_weighted_residual_sum / action_counts

        # Apply frenchel_dual in a vectorized way
        semi_a_weighted_residual_list = self.semi_q_alpha * frenchel_dual(self.f_name, semi_a_weighted_residual_mean / self.semi_q_alpha)

        # mu_dual_loss = torch.mean(torch.stack(semi_a_weighted_residual_list))
        mu_dual_loss = torch.mean(semi_a_weighted_residual_list)
        mu_linear_loss = (1 - self.discount) * torch.mean(self.mu(init_observations))
        mu_loss = mu_linear_loss + mu_dual_loss

        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        log_dict["vdice_loss/mu_loss"] = mu_loss.item()
        log_dict["value_train/mu_value"] = mu.mean().item()
        # log_dict["value_train/mu_value"] = self.mu(observations).mean().item()

    def _update_policy(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):

        semi_s_weight, semi_a_weight, true_sa_weight = self.get_weights(observations, 
                                                                        next_observations, 
                                                                        actions, 
                                                                        rewards, 
                                                                        terminals)


        log_dict["vdice_loss/a_weight"] = semi_a_weight.mean().item()
        log_dict["vdice_loss/s_weight"] = semi_s_weight.mean().item()


        

        if self.total_it >= 3e5:
            policy_out = self.semi_s_actor(observations)
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            policy_loss = torch.mean(semi_s_weight * bc_losses)
            self.semi_s_actor_optimizer.zero_grad()
            policy_loss.backward()
            self.semi_s_actor_optimizer.step()
            self.semi_s_actor_lr_schedule.step()
            log_dict["vdice_loss/semi_s_policy_loss"] = policy_loss.item()

            policy_out = self.semi_sa_actor_and(observations)
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            policy_loss = torch.mean(semi_s_weight * bc_losses)
            self.semi_sa_actor_and_optimizer.zero_grad()
            policy_loss.backward()
            self.semi_sa_actor_and_optimizer.step()
            self.semi_sa_actor_and_lr_schedule.step()
            log_dict["vdice_loss/semi_sa_policy_loss"] = policy_loss.item()

            policy_out = self.semi_sa_actor_or(observations)
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            semi_a_or_s_weight = torch.max(semi_a_weight, semi_s_weight)
            policy_loss = torch.mean(semi_a_or_s_weight * bc_losses)
            self.semi_sa_actor_or_optimizer.zero_grad()
            policy_loss.backward()
            self.semi_sa_actor_or_optimizer.step()
            self.semi_sa_actor_or_lr_schedule.step()
            log_dict["vdice_loss/semi_sa_or_policy_loss"] = policy_loss.item()
        
        if self.total_it < 3e5:
            policy_out = self.semi_a_actor(observations)
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
            policy_loss = torch.mean(semi_a_weight * bc_losses)
            self.semi_a_actor_optimizer.zero_grad()
            policy_loss.backward()
            self.semi_a_actor_optimizer.step()
            self.semi_a_actor_lr_schedule.step()
            log_dict["vdice_loss/semi_a_policy_loss"] = policy_loss.item()

        policy_out = self.true_sa_actor(observations)
        bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        policy_loss = torch.mean(true_sa_weight * bc_losses)
        self.true_sa_actor_optimizer.zero_grad()
        policy_loss.backward()
        self.true_sa_actor_optimizer.step()
        self.true_sa_actor_lr_schedule.step()
        log_dict["vdice_loss/true_sa_policy_loss"] = policy_loss.item()

        # policy_out = self.bc_actor(observations)
        # bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        # policy_loss = torch.mean(bc_losses)
        # self.bc_actor_optimizer.zero_grad()
        # policy_loss.backward()
        # self.bc_actor_optimizer.step()
        # self.bc_actor_lr_schedule.step()
        # log_dict["vdice_loss/bc_policy_loss"] = policy_loss.item()
        

    
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            actions_list,
            next_states_list,
            dones_list,
        ) = batch
        log_dict = {}
        flags = torch.ones_like(rewards)

        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        # TODO: init_observations is not the same always
        init_observations = torch.as_tensor(np.array([[12.5, 4.5]] * observations.shape[0], dtype=np.float32), device=self.device)


        if self.total_it < 3e5:
            # Update V function
            semi_residual, true_residual = self._update_v(observations, 
                                                        actions, 
                                                        next_observations, 
                                                        rewards, 
                                                        dones, 
                                                        init_observations, 
                                                        log_dict)
            # Update Q function
            self._update_q(observations, 
                        actions, 
                        next_observations, 
                        rewards, 
                        dones, 
                        log_dict)

        # if self.total_it == 600001:
        #     self.semi_q = TwinQ(2, 2, layernorm=False, hidden_dim=256).to(self.device)
        #     self.semi_q_optimizer = torch.optim.Adam(self.semi_q.parameters(), lr=3e-4)


        self._update_mu(observations,
                            actions,
                            next_observations, 
                            dones, 
                            init_observations,
                            actions_list,
                            next_states_list,
                            dones_list,
                            log_dict)

        if self.total_it >= 3e5:
            
            # # Update U function
            # s_a_weight = self._update_U(semi_residual, 
            #                             observations, 
            #                             next_observations, 
            #                             dones, 
            #                             log_dict)

            # # Update Mu function
            self._update_mu(observations,
                            actions,
                            next_observations, 
                            dones, 
                            init_observations,
                            actions_list,
                            next_states_list,
                            dones_list,
                            log_dict)


            # Update semi Q function
            # self._update_semi_q(observations,
            #                     actions,
            #                     next_observations,
            #                     rewards,
            #                     dones,
            #                     init_observations,
            #                     log_dict)
                                

            
        # Update actor
        self._update_policy(observations, 
                            next_observations, 
                            actions, 
                            rewards, 
                            dones, 
                            log_dict,)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        nn_state_dict = {
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
            "semi_q": self.semi_q.state_dict(),
            "semi_q_optimizer": self.semi_q_optimizer.state_dict(),

            "semi_sa_actor_and": self.semi_sa_actor_and.state_dict(),
            "semi_sa_actor_and_optimizer": self.semi_sa_actor_and_optimizer.state_dict(),
            "semi_sa_actor_and_lr_schedule": self.semi_sa_actor_and_lr_schedule.state_dict(),

            "semi_sa_actor_or": self.semi_sa_actor_or.state_dict(),
            "semi_sa_actor_or_optimizer": self.semi_sa_actor_or_optimizer.state_dict(),
            "semi_sa_actor_or_lr_schedule": self.semi_sa_actor_or_lr_schedule.state_dict(),

            "semi_s_actor": self.semi_s_actor.state_dict(),
            "semi_s_actor_optimizer": self.semi_s_actor_optimizer.state_dict(),
            "semi_s_actor_lr_schedule": self.semi_s_actor_lr_schedule.state_dict(),

            "semi_a_actor": self.semi_a_actor.state_dict(),
            "semi_a_actor_optimizer": self.semi_a_actor_optimizer.state_dict(),
            "semi_a_actor_lr_schedule": self.semi_a_actor_lr_schedule.state_dict(),

            "true_sa_actor": self.true_sa_actor.state_dict(),
            "true_sa_actor_optimizer": self.true_sa_actor_optimizer.state_dict(),
            "true_sa_actor_lr_schedule": self.true_sa_actor_lr_schedule.state_dict(),

            "total_it": self.total_it,
        }

        nn_state_dict["train_config"] = asdict(config)
        return nn_state_dict

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
        # self.semi_q.load_state_dict(state_dict["semi_q"])
        # self.semi_q_optimizer.load_state_dict(state_dict["semi_q_optimizer"])

        self.semi_sa_actor_and.load_state_dict(state_dict["semi_sa_actor_and"])
        self.semi_sa_actor_and_optimizer.load_state_dict(state_dict["semi_sa_actor_and_optimizer"])
        # self.semi_sa_actor_and_lr_schedule.load_state_dict(state_dict["semi_sa_actor_and_lr_schedule"])

        self.semi_sa_actor_or.load_state_dict(state_dict["semi_sa_actor_or"])
        self.semi_sa_actor_or_optimizer.load_state_dict(state_dict["semi_sa_actor_or_optimizer"])
        # self.semi_sa_actor_or_lr_schedule.load_state_dict(state_dict["semi_sa_actor_or_lr_schedule"])

        self.semi_s_actor.load_state_dict(state_dict["semi_s_actor"])
        self.semi_s_actor_optimizer.load_state_dict(state_dict["semi_s_actor_optimizer"])
        # self.semi_s_actor_lr_schedule.load_state_dict(state_dict["semi_s_actor_lr_schedule"])

        self.semi_a_actor.load_state_dict(state_dict["semi_a_actor"])
        self.semi_a_actor_optimizer.load_state_dict(state_dict["semi_a_actor_optimizer"])
        # self.semi_a_actor_lr_schedule.load_state_dict(state_dict["semi_a_actor_lr_schedule"])

        self.true_sa_actor.load_state_dict(state_dict["true_sa_actor"])
        self.true_sa_actor_optimizer.load_state_dict(state_dict["true_sa_actor_optimizer"])
        # self.true_sa_actor_lr_schedule.load_state_dict(state_dict["true_sa_actor_lr_schedule"])

        self.total_it = int(state_dict["total_it"])

    def get_weights(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
    ):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            semi_v = self.semi_v(observations)
            adv = target_q - semi_v
            semi_a_weight = f_prime_inverse(self.f_name, adv)
            
            mu = self.mu(observations)
            mu_next = self.mu(next_observations)
            mu_residual = (1.0 - terminals.float()) * self.discount * mu_next - mu
            semi_s_weight = f_prime_inverse(self.f_name, semi_a_weight * mu_residual)




            true_v = self.true_v(observations)
            true_v_next = self.true_v(next_observations)
            true_residual = rewards + (1.0 - terminals.float()) * self.discount * true_v_next - true_v
            true_residual = true_residual / self.true_dice_alpha
            true_sa_weight = f_prime_inverse(self.f_name, true_residual)
        
        return semi_s_weight, semi_a_weight, true_sa_weight
    
    def get_value(self,
                    observations=None,
                    env=None):
        if observations is None:
            state_map = np.empty(env.walls.shape + (2,), dtype=np.float32)
            for x in range(env.walls.shape[1]):
                for y in range(env.walls.shape[0]):
                    state_map[y, x] = np.array([y+0.5, x+0.5])
            observations = torch.as_tensor(state_map.reshape(-1, 2), device=self.device, dtype=torch.float32)
            with torch.no_grad():
                state_value = self.U(observations)
                action_state_value = self.semi_v(observations)
                true_state_value = self.true_v(observations)
                mu_state_value = self.mu(observations)
            state_value = state_value.cpu().numpy().reshape(env.walls.shape[0], env.walls.shape[1],)
            action_state_value = action_state_value.cpu().numpy().reshape(env.walls.shape[0], env.walls.shape[1],)
            true_state_value = true_state_value.cpu().numpy().reshape(env.walls.shape[0], env.walls.shape[1],)
            mu_state_value = mu_state_value.cpu().numpy().reshape(env.walls.shape[0], env.walls.shape[1],)
            # find the min valve of the state value which is not wall
            state_value[env.walls] = np.min(state_value[env.walls == 0])
            action_state_value[env.walls] = np.min(action_state_value[env.walls == 0])
            true_state_value[env.walls] = np.min(true_state_value[env.walls == 0])
            mu_state_value[env.walls] = np.min(mu_state_value[env.walls == 0])
        else:
            with torch.no_grad():
                state_value = self.U(observations)
                action_state_value = self.semi_v(observations)
                true_state_value = self.true_v(observations)
                mu_state_value = self.mu(observations)
                state_value = state_value.cpu().numpy()
                action_state_value = action_state_value.cpu().numpy()
                true_state_value = true_state_value.cpu().numpy()
                mu_state_value = mu_state_value.cpu().numpy()

        return state_value, action_state_value, true_state_value, mu_state_value

def trainer_init(config: TrainConfig, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)

        os.makedirs(config.checkpoints_path+"/model", exist_ok=True)

        os.makedirs(config.checkpoints_path+"/gif/semi_s_and_a", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/gif/semi_s_or_a", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/gif/semi_s", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/gif/semi_a", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/gif/true_s_and_a", exist_ok=True)
        os.makedirs(config.checkpoints_path+"/gif/bc", exist_ok=True)


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

    semi_q = SingleQ(state_dim, action_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    semi_q_optimizer = torch.optim.Adam(semi_q.parameters(), lr=config.vf_lr)

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

        "semi_q": semi_q,
        "semi_q_optimizer": semi_q_optimizer,

        "mu_network": mu_network,
        "mu_optimizer": mu_optimizer,
        "U_network": U_network,
        "U_optimizer": U_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # VDICE
        "vdice_type": config.vdice_type,
        "semi_dice_lambda": config.semi_dice_lambda,
        "true_dice_alpha": config.true_dice_alpha,
        "max_steps": config.max_timesteps,
        "semi_q_alpha": config.semi_q_alpha,
    }

    print("---------------------------------------")
    print(f"Training VDICE, Env: {config.env_1}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = VDICE(**kwargs)
    return trainer


def create_dataset(config: TrainConfig):

    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                                goal=np.array([4.5, 12.5], dtype=np.float32), 
                                goal_radius=0.8,
                                env_name=config.env_name,
                                reward_type=config.reward_type)
    dataset = np.load("dataset.npy", allow_pickle=True)
    # import pdb; pdb.set_trace()
    expert_dataset = np.load("expert_dataset.npy", allow_pickle=True)

    expert_idx = np.random.choice(np.arange(int(dataset["observations"].shape[0])), int(config.percent_expert * dataset["observations"].shape[0]), 
                                  replace=False)
    dataset["observations"][expert_idx] = expert_dataset["observations"][expert_idx]
    dataset["actions"][expert_idx] = expert_dataset["actions"][expert_idx]
    dataset["rewards"][expert_idx] = expert_dataset["rewards"][expert_idx]
    dataset["next_observations"][expert_idx] = expert_dataset["next_observations"][expert_idx]
    dataset["terminals"][expert_idx] = expert_dataset["terminals"][expert_idx]


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset["id"] = np.arange(dataset["actions"].shape[0])

    if config.normalize_reward:
        modify_reward(dataset, config.env_1)
        modify_reward(expert_dataset, config.env_1)

    # modify reward
    for i in range(dataset["observations"].shape[0]):
        obs = dataset["observations"][i]
        if np.linalg.norm(obs - env._goal) < env._goal_radius:
            dataset["reward"][i] = 1

    for i in range(expert_dataset["observations"].shape[0]):
        obs = expert_dataset["observations"][i]
        if np.linalg.norm(obs - env._goal) < env._goal_radius:
            expert_dataset["reward"][i] = 1

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


    return dataset, expert_dataset, state_dim, action_dim, env, replay_buffer

def eval_policy(actor, global_step, gif_dir, device, wandb, name):
    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8,
                               env_name=config.env_name,
                               reward_type=config.reward_type)
    
    actor.eval()
    images = []
    traj = []
    episode_lengths = []
    count_success = []
    for i in range(1):
        episode_return = 0.0
        episode_length = 0
        traj.append([])
        done = False
        obs, _ = env.reset()
        traj[-1].append(obs)
        images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
        while not done:
            with torch.no_grad():
                mean = actor.act(torch.Tensor([obs]).to(device), device=device)
            obs, reward, done, trunc, info = env.step(mean)
            images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
            traj[-1].append(obs)
            episode_return += reward
            episode_length += 1
            if done and info["success"]:
                count_success.append(1)
            elif done and not info["success"]:
                count_success.append(0)
        episode_lengths.append(episode_length)

    actor.train()
    # save images into gif
    imageio.mimsave(gif_dir + "/" +str(global_step) + ".gif", images, fps=10)

    success_rate = np.mean(count_success)
    print(name+f" Success rate: {success_rate}")
    return success_rate, np.mean(episode_lengths), traj 

def get_weights(dataset, trainer):

    semi_s_weight = []
    semi_a_weight = []
    semi_s_or_a_weight = []
    semi_s_and_a_weight = []
    true_sa_weight = []

    for i in range(0,len(dataset["observations"]),32768):
        end_idx = min(i+32768,len(dataset["observations"]))

        obs = torch.tensor(dataset["observations"][i:end_idx], dtype=torch.float32).to(trainer.device)
        next_obs = torch.tensor(dataset["next_observations"][i:end_idx], dtype=torch.float32).to(trainer.device)
        actions = torch.tensor(dataset["actions"][i:end_idx], dtype=torch.float32).to(trainer.device)
        rewards = torch.tensor(dataset["rewards"][i:end_idx], dtype=torch.float32).to(trainer.device)
        terminals = torch.tensor(dataset["terminals"][i:end_idx], dtype=torch.float32).to(trainer.device)

        semi_s, semi_a, true_sa = trainer.get_weights(obs, 
                                                        next_obs, 
                                                        actions, 
                                                        rewards, 
                                                        terminals)
        semi_s_weight.append(semi_s.cpu().numpy())
        semi_a_weight.append(semi_a.cpu().numpy())
        semi_s_or_a_weight.append(torch.max(semi_s, semi_a).cpu().numpy())
        semi_s_and_a_weight.append((semi_s).cpu().numpy())
        true_sa_weight.append(true_sa.cpu().numpy())

    semi_s_weight = np.hstack(semi_s_weight)
    semi_s_weight = semi_s_weight > 0
    
    semi_a_weight = np.hstack(semi_a_weight)
    semi_a_weight = semi_a_weight > 0
    
    semi_s_or_a_weight = np.hstack(semi_s_or_a_weight)
    semi_s_or_a_weight = semi_s_or_a_weight > 0
    
    semi_s_and_a_weight = np.hstack(semi_s_and_a_weight)
    semi_s_and_a_weight = semi_s_and_a_weight > 0

    true_sa_weight = np.hstack(true_sa_weight)
    true_sa_weight = true_sa_weight > 0


    return semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight

def draw_traj(weights, dataset, env, save_path=None, trajectories=None, values=None):
    # select the observation in dataset with weights > 0
    selected_obs = dataset["observations"][weights]
    selected_next_obs = dataset["next_observations"][weights]
    selected_terminals = dataset["terminals"][weights]
    # selected_traj_img = env.get_env_frame_with_selected_traj(obs=selected_obs, 
    #                                                         next_obs=selected_next_obs,
    #                                                         terminals=selected_terminals,
    #                                                         trajectories=trajectories,
    #                                                         save_path=save_path)

    selected_traj_img = env.get_env_frame_with_selected_traj_plt(obs=selected_obs, 
                                                            next_obs=selected_next_obs,
                                                            terminals=selected_terminals,
                                                            trajectories=trajectories,
                                                            values=values,
                                                            save_path=save_path)
    
    return selected_traj_img

def load_checkpoint(config):

    # read parameters from yaml file into config
    with open(config.load_yaml, "r") as stream:
        try:
            yaml_config = yaml.safe_load(stream)
            for key, value in yaml_config.items():
                if key != "project" and \
                key != "checkpoints_path" and \
                key != "load_model" and \
                key != "load_yaml" and \
                key != "semi_q_alpha" and \
                    key != "alg":
                    setattr(config, key, value)
        except yaml.YAMLError as exc:
            print(exc)

    dataset, expert_dataset, state_dim, action_dim, env, replay_buffer = create_dataset(config)
    semi_trainer = trainer_init(config, env)
    semi_trainer.load_state_dict(torch.load(config.load_model))

    # config.batch_size = 64
    return semi_trainer, dataset, expert_dataset, state_dim, action_dim, env, replay_buffer


def train():

    if config.load_model != "":
        semi_trainer, dataset, expert_dataset, state_dim, action_dim, env, replay_buffer = load_checkpoint(config)
    else:
        dataset, expert_dataset, state_dim, action_dim, env, replay_buffer = create_dataset(config)
        semi_trainer = trainer_init(config, env)


    wandb_init(asdict(config))
    t = 0

    sample_size = min(2000, len(expert_dataset["observations"]))
    temp_expert_dataset = {}
    temp_expert_dataset["observations"] = expert_dataset["observations"][:sample_size]
    temp_expert_dataset["actions"] = expert_dataset["actions"][:sample_size]
    temp_expert_dataset["rewards"] = expert_dataset["rewards"][:sample_size]
    temp_expert_dataset["next_observations"] = expert_dataset["next_observations"][:sample_size]
    temp_expert_dataset["terminals"] = expert_dataset["terminals"][:sample_size]

    sample_size = min(2000, len(dataset["observations"]))
    temp_dataset = {}
    temp_dataset["observations"] = dataset["observations"][:sample_size]
    temp_dataset["actions"] = dataset["actions"][:sample_size]
    temp_dataset["rewards"] = dataset["rewards"][:sample_size]
    temp_dataset["next_observations"] = dataset["next_observations"][:sample_size]
    temp_dataset["terminals"] = dataset["terminals"][:sample_size]

    if config.checkpoints_path is not None:
        weights = np.ones(sample_size, dtype=bool)
        save_dir = config.checkpoints_path + "/selected_traj/"
        os.makedirs(save_dir, exist_ok=True)
        full_traj = draw_traj(weights, temp_dataset, env, save_path = save_dir + "/full.png")
        wandb.log({"selected_traj/" + "full_traj": wandb.Image(full_traj),
                   "selected_traj_step" : semi_trainer.total_it,
                   })
        
    while t < int(config.max_timesteps):
        

        batch = replay_buffer.sample(config.batch_size, all_actions=True)
        batch = [b.to(config.device) if isinstance(b, torch.Tensor) else b for b in batch]
        semi_log_dict = semi_trainer.train(batch)
        semi_log_dict["vdice_step"] = semi_trainer.total_it
        semi_log_dict["value_step"] = semi_trainer.total_it

        wandb.log(semi_log_dict,)

        if (t + 1) % config.eval_freq == 0:
            policy_log_dict = {}
            policy_log_dict["policy_train/semi_s_and_a_perform"], \
            policy_log_dict["policy_train/semi_s_and_a_epi_len"], \
            semi_s_and_a_traj = eval_policy(semi_trainer.semi_sa_actor_and, semi_trainer.total_it, config.checkpoints_path+"/gif/semi_s_and_a", config.device, wandb, "semi_s_and_a_perform")
            
            policy_log_dict["policy_train/semi_s_or_a_perform"], \
            policy_log_dict["policy_train/semi_s_or_a_epi_len"], \
            semi_s_or_a_traj = eval_policy(semi_trainer.semi_sa_actor_or, semi_trainer.total_it, config.checkpoints_path+"/gif/semi_s_or_a", config.device, wandb, "semi_s_or_a_perform")
            
            policy_log_dict["policy_train/semi_s_perform"], \
            policy_log_dict["policy_train/semi_s_epi_len"], \
            semi_s_traj  = eval_policy(semi_trainer.semi_s_actor, semi_trainer.total_it, config.checkpoints_path+"/gif/semi_s", config.device, wandb, "semi_s_perform")
            
            policy_log_dict["policy_train/semi_a_perform"], \
            policy_log_dict["policy_train/semi_a_epi_len"], \
            semi_a_traj = eval_policy(semi_trainer.semi_a_actor, semi_trainer.total_it, config.checkpoints_path+"/gif/semi_a", config.device, wandb, "semi_a_perform")
            
            policy_log_dict["policy_train/true_s_and_a_perform"], \
            policy_log_dict["policy_train/true_s_and_a_epi_len"], \
            true_s_and_a_traj = eval_policy(semi_trainer.true_sa_actor, semi_trainer.total_it, config.checkpoints_path+"/gif/true_s_and_a", config.device, wandb, "true_s_and_a_perform")
            
            policy_log_dict["policy_train/bc_perform"], \
            policy_log_dict["policy_train/bc_epi_len"], \
            bc_traj = eval_policy(semi_trainer.bc_actor, semi_trainer.total_it, config.checkpoints_path+"/gif/bc", config.device, wandb, "bc_perform")

            policy_log_dict["policy_step"] = semi_trainer.total_it
            wandb.log(policy_log_dict,)
            print("==============================")

        if (t + 1) % config.save_freq == 0:
            if config.checkpoints_path is not None:
                torch.save(
                    semi_trainer.state_dict(),
                    os.path.join(config.checkpoints_path+"/model", f"checkpoint_{t}.pt"),
                )
                
                semi_s_state_value, semi_a_state_value, true_s_and_a_state_value, mu_state_value = semi_trainer.get_value(env=env)
                semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight = get_weights(temp_dataset, semi_trainer)
                weights_dict = {
                    "semi_s": semi_s_weight,
                    "semi_a": semi_a_weight,
                    "semi_s_or_a": semi_s_or_a_weight,
                    "semi_s_and_a": semi_s_and_a_weight,
                    "true_s_and_a": true_sa_weight,
                    "bc": np.ones_like(semi_s_weight)
                }
                for key, weights in weights_dict.items():
                    save_dir = config.checkpoints_path + "/selected_traj/" + key
                    os.makedirs(save_dir, exist_ok=True)
                    values = None
                    if key == "semi_s" or key == "semi_a" or key == "true_s_and_a":
                        values = locals()[key+"_state_value"]
                    selected_img  = draw_traj(weights, temp_dataset, env, save_path = save_dir + "/" + str(semi_trainer.total_it) + ".png", trajectories=locals()[key+"_traj"], values=values)
                    weights_dict[key] = selected_img
                
                wandb.log({"selected_traj/" + "semi_s": wandb.Image(weights_dict["semi_s"]),
                           "selected_traj/" + "semi_a": wandb.Image(weights_dict["semi_a"]),
                           "selected_traj/" + "semi_s_or_a": wandb.Image(weights_dict["semi_s_or_a"]),
                           "selected_traj/" + "semi_s_and_a": wandb.Image(weights_dict["semi_s_and_a"]),
                           "selected_traj/" + "true_s_and_a": wandb.Image(weights_dict["true_s_and_a"]),
                           "selected_traj/" + "bc": wandb.Image(weights_dict["bc"]),
                           "selected_traj_step" : semi_trainer.total_it,
                           },)
            

                semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight = get_weights(temp_expert_dataset, semi_trainer)
                weights_dict = {
                    "semi_s": semi_s_weight,
                    "semi_a": semi_a_weight,
                    "semi_s_or_a": semi_s_or_a_weight,
                    "semi_s_and_a": semi_s_and_a_weight,
                    "true_s_and_a": true_sa_weight
                }
                for key, weights in weights_dict.items():
                    save_dir = config.checkpoints_path + "/selected_expert_traj/" + key
                    os.makedirs(save_dir, exist_ok=True)
                    selected_img  = draw_traj(weights, temp_expert_dataset, env, save_path = save_dir + "/" + str(semi_trainer.total_it) + ".png")
                    weights_dict[key] = selected_img
                
                wandb.log({"selected_expert_traj/" + "semi_s": wandb.Image(weights_dict["semi_s"]),
                           "selected_expert_traj/" + "semi_a": wandb.Image(weights_dict["semi_a"]),
                           "selected_expert_traj/" + "semi_s_or_a": wandb.Image(weights_dict["semi_s_or_a"]),
                           "selected_expert_traj/" + "semi_s_and_a": wandb.Image(weights_dict["semi_s_and_a"]),
                           "selected_expert_traj/" + "true_s_and_a": wandb.Image(weights_dict["true_s_and_a"]),
                           "selected_expert_traj_step" : semi_trainer.total_it,
                           },)

        t += 1


if __name__ == "__main__":
    config = pyrallis.parse(config_class=TrainConfig)
    train()