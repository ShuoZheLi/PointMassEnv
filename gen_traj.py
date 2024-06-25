import random
import numpy as np
from PointMassEnv import PointMassEnv

def get_random_offline_trajectory(start_pos, goal, traj_num):


    traj_count = 0
    observations = []
    actions = []
    rewards = []
    terminals = []
    traj = []
    

    while traj_count < traj_num:
        action_noise = 0.8
        traj.append([])

        for i in range(len(start_pos)):
            env = PointMassEnv(start=np.array(start_pos[i], dtype=np.float32),)
            env.goal = np.array(goal[-1], dtype=np.float32)
            obs = env.reset()
            # img = env.render()
            sub_goal = np.array(goal[i], dtype=np.float32)
            # observations.append(obs)
            reached_sub_goal = False


            while not reached_sub_goal:
                observations.append(obs)
                traj[traj_count].append(obs)
                vec = sub_goal - obs
                action = vec / np.linalg.norm(vec) / 4
                action = np.random.normal(action, action_noise)
                actions.append(action)
                obs, reward, _, _ = env.step(action)
                rewards.append(reward)
                # print('reward:', reward)
                # observations.append(obs)
                # img = env.render()
                # print(img)
                if np.linalg.norm(sub_goal - obs) < 0.8:
                    # print('Success')
                    terminals.append(True)
                    # obs = env.reset()
                    action_noise -= 0.2
                    reached_sub_goal = True
                else:
                    terminals.append(False)
        
        traj[traj_count] = np.array(traj[traj_count])
        traj_count += 1
        print('Trajectory:', traj_count)
        print('Observations num:', len(observations))
        

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    terminals = np.array(terminals)
    next_observations = np.roll(observations, -1, axis=0)

    traj = np.array(traj)


    # make a offline rl dataset in the form of (obs, act, rew, next_observations, terminals)
    dataset = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals
    }


    # save the dataset
    np.save('toy_dataset.npy', dataset)
    np.save('traj.npy', traj)
    # import pdb; pdb.set_trace()

    

    
            

if __name__ == '__main__':

    np.random.seed(10)
    random.seed(10)

    # start_pos = [[15, 3], [9.5, 5.5], [3.5, 9.5]]
    # goal = [[9.5, 5.5], [3.5, 9.5], [4.5, 15.5]]

    start_pos = [[12.5, 4.5]]
    goal = [[4.5, 12.5]]


    get_random_offline_trajectory(start_pos=start_pos, goal=goal, traj_num=10)