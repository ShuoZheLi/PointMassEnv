import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import offline_RL.vdice_correct_filter
from offline_RL.vdice_correct_filter import VDICE, PointMassEnv, trainer_init, get_weights, draw_traj, modify_reward
import imageio
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from dataclasses import asdict, dataclass



@dataclass
class TrainConfig:
    #############################
    ######### Experiment ########
    #############################
    seed: int = 100
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    save_freq: int = int(5e3)  # How often (time steps) save the model
    update_freq: int = int(1.5e5)  # How often (time steps) we update the model
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(6e6)  # Max time steps to run environment
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
    env_name: str = "EmptyRoom"
    reward_type: str = "sparse"
    percent_expert: float = 0
    discrete_action: bool = False


def eval_policy(actor, global_step, gif_dir, device, wandb, name):
    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8,
                               env_name=config.env_name,
                               reward_type=config.reward_type)

    # env = PointMassEnv(start=np.array([1, 4.5], dtype=np.float32), 
    #                            goal=np.array([4.5, 12.5], dtype=np.float32), 
    #                            goal_radius=0.8,
    #                            env_name=config.env_name,
    #                            reward_type=config.reward_type)
    
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
            print("obs: ", obs)
            print("act: ", mean)
            print()
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
    imageio.mimsave(gif_dir + "/" +str(global_step) + ".gif", images, fps=2)

    success_rate = np.mean(count_success)
    print(name+f" Success rate: {success_rate}")
    return success_rate, np.mean(episode_lengths), traj 

if __name__ == "__main__":
    config = TrainConfig()

    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8,
                               env_name=config.env_name,
                               reward_type=config.reward_type)

    vdice = trainer_init(config, env)


    vdice.load_state_dict(torch.load("discrete_empty_room_offline_mini/discrete_empty_room_offline_mini_normalize_reward_True_true_dice_alpha_1_discount_0_99_semi_dice_lambda_0_5_seed_100_percent_expert_0_eval_freq_500_save_freq_500_batch_size_128/test-antmaze-umaze-v2-fa7ace7b/model/checkpoint_81499.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_policy(vdice.semi_a_actor, 0, "/data/shuozhe/discrete_toy", device, None, "name")

    exit()

    dataset = np.load('dataset.npy', allow_pickle=True)

    dataset["observations"] = np.round(dataset["observations"] * 10) / 10
    dataset["next_observations"] = np.round(dataset["next_observations"] * 10) / 10

    dataset["observations"] = dataset["observations"][-100_000:]
    dataset["actions"] = dataset["actions"][-100_000:]
    dataset["rewards"] = dataset["rewards"][-100_000:]
    dataset["next_observations"] = dataset["next_observations"][-100_000:]
    dataset["terminals"] = dataset["terminals"][-100_000:]
    # dataset["success"] = dataset["success"][:10000]

    # import pdb; pdb.set_trace()

    if config.normalize_reward:
        modify_reward(dataset, config.env_1)
    # import pdb; pdb.set_trace()
    semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight = get_weights(dataset, vdice)
    
    draw_traj(semi_a_weight, dataset, env, save_path = "/data/shuozhe/discrete_toy" + "/a_w_whole_dataset.png")
    draw_traj(np.ones_like(semi_a_weight), dataset, env, save_path = "/data/shuozhe/discrete_toy" + "/full_dataset.png")
