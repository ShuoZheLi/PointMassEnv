import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from online_RL.sac import Actor, discrete_action
import torch
import numpy as np
from PointMassEnv import PointMassEnv
import gymnasium as gym
import random
from typing import Optional
import pickle



def make_env():
    def thunk():
        return PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8, 
                               env_name="EmptyRoom")
    return thunk


def set_seed(
    seed: int, env: Optional[gym.Env] = None
):
    if env is not None:
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def gen_traj(actor, env, device, traj_dir, traj_num, transition_num, discrete=False):

    dataset = {}
    dataset["observations"] = []
    dataset["actions"] = []
    dataset["rewards"] = []
    dataset["next_observations"] = []
    dataset["terminals"] = []
    dataset["success"] = []


    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8, 
                               env_name="EmptyRoom")
    actor.eval()
    images = []
    count_success = 0
    transition_count = 0


    for i in range(traj_num):

        obs_array = []
        action_array = []
        reward_array = []
        next_obs_array = []
        terminal_array = []
        success_array = []

        done = False
        obs, _ = env.reset()
        # images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
        epi_len = 1
        while not done:
            with torch.no_grad():
                action, log_prob, mean = actor.get_action(torch.Tensor([obs]).to(device))
            mean = mean.cpu().numpy()
            if discrete:
                mean = discrete_action(mean)
                if np.random.rand() < 0.83:
                    mean[0][0] = np.random.choice([-1, 0, 1])
                    mean[0][1] = np.random.choice([-1, 0, 1])
            else:
                mean = np.random.normal(mean, 20 / epi_len)
            mean = mean[0]

            obs_array.append(obs)
            action_array.append(mean)
            # dataset["observations"].append(obs)
            # dataset["actions"].append(mean)
            obs, reward, done, trunc, info = env.step(mean)
            # dataset["rewards"].append(reward)
            # dataset["terminals"].append(done)
            reward_array.append(reward)
            next_obs_array.append(obs)
            terminal_array.append(done)

            transition_count += 1
            epi_len += 1
            if done and info["success"]:
                count_success += 1
                # dataset["success"].append(True)
                print("success length: ", epi_len)
                success_array.append(True)
                if epi_len != 8:
                    dataset["observations"] += obs_array
                    dataset["actions"] += action_array
                    dataset["rewards"] += reward_array
                    dataset["next_observations"] += next_obs_array
                    dataset["terminals"] += terminal_array
                    success_array = [True] * len(success_array)
                    dataset["success"] += success_array
                else:
                    print("rarely good !!!!")
            elif done:
                dataset["observations"] += obs_array
                dataset["actions"] += action_array
                dataset["rewards"] += reward_array
                dataset["next_observations"] += next_obs_array
                dataset["terminals"] += terminal_array
                dataset["success"] += success_array
                # pass
            else:
                # dataset["success"].append(False)
                success_array.append(False)
                

        # print success rate
        print("success count: ", count_success)
        print("num of trans: ", transition_count)
        print()
        if transition_count > transition_num:
            break

    # dataset["next_observations"] = dataset["observations"][1:] + [obs]

    dataset["observations"] = np.array(dataset["observations"], dtype=np.float32)
    dataset["actions"] = np.array(dataset["actions"], dtype=np.float32)
    dataset["rewards"] = np.array(dataset["rewards"], dtype=np.float32)
    dataset["next_observations"] = np.array(dataset["next_observations"], dtype=np.float32)
    dataset["terminals"] = np.array(dataset["terminals"], dtype=np.float32)
    dataset["success"] = np.array(dataset["success"], dtype=np.float32)

    # import pdb; pdb.set_trace()
    # save the trajectory
    with open(traj_dir, 'wb') as f:
        pickle.dump(dataset, f)


# load the actor and generate the trajectory
if __name__ == "__main__":

    envs = gym.vector.SyncVectorEnv([make_env()])
    set_seed(0, envs,)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(envs).to(device)
    actor.load("dataset/EmptyRoom/uniform/expert_actor.pth")
    gen_traj(actor, envs, device, "dataset.npy", 90000000, transition_num=5000, discrete=True)


