import gym
import numpy as np
import random


WALLS = {
        'IndianWell':
                np.array([[0, 0, 0, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 0], 
                          [0, 0, 1, 1, 0, 0],
                          [0, 0, 1, 1, 0, 0], 
                          [0, 0, 1, 1, 0, 0], 
                          [0, 0, 1, 1, 0, 0],
                          [0, 0, 1, 1, 0, 0], 
                          [0, 0, 1, 1, 0, 0], 
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]),
        'IndianWell2':
                np.array([[0, 0, 0, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 0], 
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0], 
                          [0, 1, 1, 0, 0, 0], 
                          [0, 1, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0], 
                          [0, 1, 1, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]),
        'IndianWell3':
                np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]]),
        'DrunkSpider':
                np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]),
        'DrunkSpiderShort':
                np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]),
        'FourRooms':
                np.array([
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], ## tunnel
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1], ## tunnel
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], ## tunnel
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                ]),
}


def resize_walls(walls, factor):
    """Increase the environment by rescaling.
    Args:
        walls: 0/1 array indicating obstacle locations.
        factor: (int) factor by which to rescale the environment.
    Returns:
        walls: rescaled walls
    """
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
    col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


class PointMassEnv(gym.Env):
    """Class for 2D navigation in PointMass environment."""

    def __init__(self,
                env_name='FourRooms',
                start=None,
                resize_factor=1,
                action_noise=0,
                start_bounds=None,
                num_substeps=50,
                ):
        """
        Args:
            env_name: environment name
            start: starting position
            resize_factor: (int) Scale the map by this factor.
            action_noise: (float) Standard deviation of noise to add to actions. Use 0
                to add no noise.
            start_bounds: starting bound
        """
        walls = env_name

        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._num_substeps = num_substeps

        ## start position
        if start_bounds is not None:
            self._start_space = gym.spaces.Box(
                    low=start_bounds[0], high=start_bounds[1], dtype=np.float32)
        else:
            self._start_space = gym.spaces.Box(
                    low=np.zeros(2), high=np.array([height, width]), dtype=np.float32)

        ## action space
        self._action_noise = action_noise
        self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        ## observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self._height, self._width]),
            dtype=np.float32
        )

        if start is None:
            self._start = None
        else:
            self._start = np.array(start, dtype=np.float32)
            assert self._is_valid_state(self._start)

        self.goal = np.array([0,0], dtype=np.float32)

        self.reset()

    ##########################
    #### helper functions ####
    ##########################

    @property
    def walls(self):
        return self._walls

    def _get_obs(self):
        return self.state.copy()

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int)
        # Round down to the nearest cell if at the boundary.
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return np.array([i, j], dtype=np.int)

    def _sample_start_state(self):
        if self._start is not None:
            state = self._start.copy()
        else:
            state = self.sample_valid_state(self._start_space)
        return state

    def sample_valid_state(self, space=None):
        space = space or self.observation_space
        i = 0
        valid = False
        while not valid:
            state = self.observation_space.sample()
            valid = self._is_valid_state(state)
            i += 1
            if i > 10:
                print(f'Warning: sampling {i} start states')
        return state

    ##########################
    #### state validation ####
    ##########################

    def _is_valid_state(self, state):
        return not self._is_out_of_bounds(state) \
               and not self._is_in_wall(state)

    def _is_out_of_bounds(self, state):
        return not self.observation_space.contains(state)

    def _is_in_wall(self, state):
        i, j = self._discretize_state(state)
        # return (self._walls[i, j] == 1)

        x, y = state


        walls = [
            # Outer boundaries
            # (0, 0, 1, 19),     # Left wall
            # (18, 0, 19, 19),   # Right wall
            # (0, 0, 19, 1),     # Top wall
            # (0, 18, 19, 19),   # Bottom wall
            
            # (9, 1, 10, 3),
            # (9, 4, 10, 12),
            # (9.25, 4, 9.75, 12),
            # (9, 13, 10, 18),

            # (1, 9, 5, 10),
            # (6, 9, 14, 10),
            # (15, 9, 18, 10),

            # (4, 9, 12, 10),


            (0, 0, 19, 1),     # Left wall
            (0, 18, 19, 19),   # Right wall
            (0, 0, 1, 19),     # Top wall
            (18, 0, 19, 19),   # Bottom wall
            
            (1, 9, 3, 10),
            (4, 9, 12, 10),
            (4, 9, 12, 10),
            (13, 9, 18, 10),

            (9, 1, 10, 5),
            (9, 6, 10, 14),
            (9, 15, 10, 18),
        ]
        
        # Check if the agent is in any of the walls
        for (x1, y1, x2, y2) in walls:
            if x1 < x < x2 and y1 < y < y2:
                return True

        
        # return (self._walls[i, j] == 1)
        return False
    
    def in_wall_check(self, state):
        x, y = state


        walls = [
            # Outer boundaries
            # (0, 0, 1, 19),     # Left wall
            # (18, 0, 19, 19),   # Right wall
            # (0, 0, 19, 1),     # Top wall
            # (0, 18, 19, 19),   # Bottom wall
            
            # (9, 1, 10, 3),
            # (9, 4, 10, 12),
            # (9.25, 4, 9.75, 12),
            # (9, 13, 10, 18),

            # (1, 9, 5, 10),
            # (6, 9, 14, 10),
            # (15, 9, 18, 10),

            (4, 9, 12, 10),

        ]
        
        # Check if the agent is in any of the walls
        for (x1, y1, x2, y2) in walls:
            if x1 < x < x2 and y1 < y < y2:
                return True



    ##########################
    ######## main api ########
    ##########################

    def reset(self, reset_args=None):
        self.state = reset_args or self._sample_start_state()
        return self._get_obs()

    def set_state(self, state):
        self.state = state.copy()

    def render(self):
        i, j = self._discretize_state(self.state)
        img = self._walls.copy()
        img[i, j] = 2
        return img

    def step(self, action):
        action = np.array(action, dtype=np.float32)

        ## apply noise
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)

        ## clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        ## take action in substeps in case we hit a wall
        dt = 1.0 / self._num_substeps

        old_state = self.state.copy()

        state, valid, substeps = self._step_repeats(action * dt, self._num_substeps)

        if not valid:
            self.state = old_state.copy()
        else:
            self.state = state.copy()

        max_dist = np.linalg.norm(self._start - self.goal)
        reward = (max_dist - np.linalg.norm(self.state - self.goal)) / max_dist

        term = False

        return self._get_obs(), reward, term, {}

    def step_discrete(self, state, action):
        ## position is index (i, j)
        position = self._discretize_state(np.array(state))
        self.state = position.copy()
        next_state, *_ = self.step(action)
        next_position = self._discretize_state(next_state)
        return next_position

    def _step_repeats(self, action, N):
        current_state = self.state.copy()
        for i in range(N):

            new_state = current_state.copy()
            new_state = new_state + action

            valid = self._is_valid_state(new_state)
            if not valid:
                break

            current_state = new_state.copy()

        return current_state, valid, i + valid

def get_random_trajectory(start_pos, goal, traj_num):
    traj_count = 0
    traj = []
    

    while traj_count < traj_num:
        traj.append([])
        action_noise = 0.8

        for i in range(len(start_pos)):
            env = PointMassEnv(start=np.array(start_pos[i], dtype=np.float32), action_noise=0)
            obs = env.reset()
            img = env.render()
            sub_goal = np.array(goal[i], dtype=np.float32)
            traj[traj_count].append(obs)
            reached_sub_goal = False

            print('end goal:', sub_goal)

            while not reached_sub_goal:
                vec = sub_goal - obs
                action = vec / np.linalg.norm(vec) / 4
                action = np.random.normal(action, action_noise)
                obs, reward, _, _ = env.step(action)
                traj[traj_count].append(obs)
                img = env.render()
                # print(img)
                if np.linalg.norm(sub_goal - obs) < 0.8:
                    print('Success')
                    # obs = env.reset()
                    action_noise -= 0.2
                    reached_sub_goal = True
        
        traj[traj_count] = np.array(traj[traj_count])
        traj_count += 1

    traj = np.array(traj)
    
    np.save('traj.npy', traj)

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
            env = PointMassEnv(start=np.array(start_pos[i], dtype=np.float32), action_noise=0)
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
                    print('Success')
                    terminals.append(True)
                    # obs = env.reset()
                    action_noise -= 0.2
                    reached_sub_goal = True
                else:
                    terminals.append(False)
        
        traj[traj_count] = np.array(traj[traj_count])
        traj_count += 1
        

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
    import pdb; pdb.set_trace()

    

    
            

if __name__ == '__main__':

    np.random.seed(10)
    random.seed(10)

    # start_pos = [[15, 3], [9.5, 5.5], [3.5, 9.5]]
    # goal = [[9.5, 5.5], [3.5, 9.5], [4.5, 15.5]]

    start_pos = [[15, 3]]
    goal = [[4.5, 15.5]]

    # get_random_trajectory(start_pos, goal, 100)

    get_random_offline_trajectory(start_pos, goal, 10)

    