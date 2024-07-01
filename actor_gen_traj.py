from sac import Actor
import torch
import numpy as np
from PointMassEnv import PointMassEnv
import gymnasium as gym
import os
import random
from typing import Optional

def make_env():
    def thunk():
        return PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                                goal=np.array([4.5, 12.5], dtype=np.float32), 
                                goal_radius=0.8)
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

def gen_traj(actor, env, device, traj_dir, traj_num, transition_num):

    dataset = {}
    dataset["observations"] = []
    dataset["actions"] = []
    dataset["rewards"] = []
    dataset["next_observations"] = []
    dataset["terminals"] = []


    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                               goal=np.array([4.5, 12.5], dtype=np.float32), 
                               goal_radius=0.8)
    traj = []
    actor.eval()
    images = []
    count_success = 0
    transition_count = 0
    for i in range(traj_num):
        traj.append([])
        done = False
        obs, _ = env.reset()
        traj[-1].append(obs)
        images.append(np.moveaxis(np.transpose(env.render()), 0, -1))
        while not done:
            with torch.no_grad():
                action, log_prob, mean = actor.get_action(torch.Tensor([obs]).to(device))
            mean = mean[0].cpu().numpy()
            mean = np.random.normal(mean, 1)
            dataset["observations"].append(obs)
            dataset["actions"].append(mean)
            obs, reward, done, trunc, info = env.step(mean)
            dataset["rewards"].append(reward)
            dataset["terminals"].append(done)
            traj[-1].append(obs)
            transition_count += 1
            if done and info["success"]:
                count_success += 1 
        # print success rate
        print("success rate: ", count_success / (i + 1))
        print("num of trans: ", transition_count)
        # if transition_count > transition_num:
        #     break

    dataset["next_observations"] = dataset["observations"][1:] + [obs]

    dataset["observations"] = np.array(dataset["observations"])
    dataset["actions"] = np.array(dataset["actions"])
    dataset["rewards"] = np.array(dataset["rewards"])
    dataset["next_observations"] = np.array(dataset["next_observations"])
    dataset["terminals"] = np.array(dataset["terminals"])
    dataset["trajectories"] = np.array(traj, dtype=object)


    # save the trajectory
    np.save(traj_dir, dataset)


# load the actor and generate the trajectory
if __name__ == "__main__":

    envs = gym.vector.SyncVectorEnv([make_env()])
    set_seed(0, envs,)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(envs).to(device)
    actor.load("runs/neg_l2_save_model__sac__200__1719560558/models/490000_actor.pth")
    gen_traj(actor, envs, device, "dataset.npy", 5, transition_num=1000)


