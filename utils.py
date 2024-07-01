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

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._experts = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        # self._experts[:n_transitions] = self._to_tensor(data["expert"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        # experts = self._experts[indices]
        # return [states, actions, rewards, next_states, dones, experts]
        return [states, actions, rewards, next_states, dones,]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)



@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
        layernorm: bool = False,
    ):
        super().__init__()
        self.layernorm = layernorm
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            linear_layer = nn.Linear(dims[i], dims[i + 1])
            nn.init.constant_(linear_layer.bias, 0.1)
            layers.append(linear_layer)
             
            layers.append(activation_fn())
            layers.append( nn.LayerNorm(dims[i + 1]) if self.layernorm else nn.Identity(),)
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        final_linear_layer = nn.Linear(dims[-2], dims[-1])
        nn.init.uniform_(final_linear_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_linear_layer.weight, -3e-3, 3e-3)
        # nn.init.constant_(final_linear_layer.bias, 0.1)
        layers.append(final_linear_layer)
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinV(nn.Module):
    def __init__(
        self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v1 = MLP(dims, squeeze_output=True)
        self.v2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return self.v1(state), self.v2(state)
        return torch.stack([self.v1(state), self.v2(state)], dim=0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # return torch.min(*self.both(state))
        return torch.min(self.both(state), dim=0)[0]


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2, layernorm: bool = False):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True, layernorm=layernorm)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2, layernorm: bool = False
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True, layernorm=layernorm)
        self.q2 = MLP(dims, squeeze_output=True, layernorm=layernorm)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))
    

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