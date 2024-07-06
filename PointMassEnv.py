import gymnasium as gym
import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt
from typing import Union, Optional
from gymnasium.spaces import Space
from PIL import Image, ImageDraw
import pickle
from WALLS import WALLS



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


# class PointMassEnv(gym.Env):
class PointMassEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """Class for 2D navigation in PointMass environment."""

    def __init__(self,
                env_name='FourRooms',
                reward_type='sparse',
                start=None,
                goal=None,
                resize_factor=1,
                action_noise=0,
                start_bounds=None,
                num_substeps=50,
                goal_radius=0.8,
                episode_length=60,
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
        self._reward_type = reward_type

        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        self._bounds = WALLS[f'{walls}Bounds']
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._num_substeps = num_substeps
        self._goal_radius = goal_radius
        self._episode_length = episode_length
        self.render_mode = 'rgb_array'

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

        if goal is None:
            self._goal = None
        else:
            self._goal = np.array(goal, dtype=np.float32)
            assert self._is_valid_state(self._goal)

        self.reset()

    ##########################
    ######## main api ########
    ##########################

    def reset(self, 
              reset_state=None, 
              seed=0,
              options: Optional[dict] = None,):
        super().reset(seed=seed)
        if reset_state is not None:
            self.state = reset_state
        else:
            self.state = self._start
        self.epi_length = 0
        return self._get_obs(), {}
    
    # def render(self):
    #     i, j = self._discretize_state(self.state)
    #     img = self._walls.copy()
    #     img[i, j] = 2
    #     return img
    
    def reward(self, state):
        reward = 0

        if self._reward_type == 'sparse':
            if self.check_success(state):
                reward = 1
        else:
            reward = - np.linalg.norm(state - self._goal)

        return reward
    
    def check_success(self, state):
        return np.linalg.norm(state - self._goal) < self._goal_radius

    def step(self, action):
        self.epi_length += 1
        action = np.array(action, dtype=np.float32)

        ## apply noise
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise)

        ## clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        ## take action in substeps in case we hit a wall
        dt = 1.0 / self._num_substeps
        old_state = self.state.copy()
        state, valid = self._step_repeats(action * dt, self._num_substeps)
        self.state = state.copy()

        # TODO: we can give penalty for hitting the wall (valid = False)
        reward = self.reward(self.state)
        term = trunc = self.check_success(self.state) or self.epi_length >= self._episode_length

        return self._get_obs(), reward, term, trunc, {'success': self.check_success(self.state), 'valid': valid}

    def render(self):
        return self.get_env_frame(self.state, self._goal)

    def seed(self, seed):
        np.random.seed(seed)

    ##########################
    #### helper functions ####
    ##########################

    @property
    def walls(self):
        return self._walls

    def set_state(self, state):
        self.state = state.copy()

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

        return current_state, valid

    ##########################
    #### state validation ####
    ##########################

    def _is_valid_state(self, state):
        return not self._is_out_of_bounds(state) \
               and not self._is_in_wall(state)

    def _is_out_of_bounds(self, state):
        return not self.observation_space.contains(state)

    def _is_in_wall(self, state):
        x, y = state
        
        # Check if the agent is in any of the walls
        for (x1, y1, x2, y2) in self._bounds:
            if x1 < x < x2 and y1 < y < y2:
                return True

        return False
    
    ##########################
    ###### better render #####
    ##########################
    def get_env_frame_2(self, pos, goal):

        # Create the plot
        plt.figure(figsize=(8, 8))
        extent = [0, self._walls.shape[1], 0, self._walls.shape[0]]
        plt.imshow(self._walls, cmap='Greys', interpolation='none', extent=extent)

        plt.scatter(pos[1], pos[0], color='blue', s=100)
        plt.scatter(goal[1], goal[0], color='red', s=100)
        

        plt.grid(color='gray', linestyle='-', linewidth=0.25)

        # change the size of the grid box to be 1x1
        plt.xticks(np.arange(0, 17, 1))
        plt.yticks(np.arange(0, 17, 1))
        plt.grid(True)

        # plt.axis('off')
        plt.title('Four-room')
        plt.gca().invert_yaxis()
        fig = plt.gcf()
        fig.canvas.draw()
        # convert canvas to image using numpy
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.fliplr(img)
        img = np.rot90(img)

        plt.figure().clear()
        plt.figure().clf()
        plt.close()
        plt.cla()
        plt.clf()

        return img

    def get_env_frame(self, pos, goal, save_path='env_frame.png'):
        
        pos = pos - 0.5
        goal = goal - 0.5

        # Create a blank image with white background
        img = Image.new('RGB', (self._walls.shape[1] * 10, self._walls.shape[0] * 10), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw walls
        for y in range(self._walls.shape[0]):
            for x in range(self._walls.shape[1]):
                if self._walls[y, x] == 1:
                    draw.rectangle([x * 10, y * 10, (x + 1) * 10, (y + 1) * 10], fill='black')

        # Draw position and goal
        draw.ellipse([(pos[1] * 10, pos[0] * 10), (pos[1] * 10 + 10, pos[0] * 10 + 10)], fill='blue', width=0.25)
        draw.ellipse([(goal[1] * 10, goal[0] * 10), (goal[1] * 10 + 10, goal[0] * 10 + 10)], fill='red', width=0.25)

        # Draw grid
        for x in range(0, self._walls.shape[1] * 10, 10):
            draw.line([(x, 0), (x, self._walls.shape[0] * 10)], fill='gray', width=1)
        for y in range(0, self._walls.shape[0] * 10, 10):
            draw.line([(0, y), (self._walls.shape[1] * 10, y)], fill='gray', width=1)

        # Save the image
        # img.save(save_path)

        img = img.resize((800, 800), Image.LANCZOS)

        # Convert PIL image to numpy array
        img_array = np.array(img)
        img_array = np.moveaxis(np.transpose(img_array), 0, -1)
        return img_array

    def get_env_frame_with_selected_traj(self, start=None, goal=None, obs=None, next_obs=None, terminals=None, save_path=None):
        
        if start is None:
            start = self._start
        
        if goal is None:
            goal = self._goal
        start = start - 0.5
        goal = goal - 0.5

        # Create a blank image with a white background
        img = Image.new('RGB', (self._walls.shape[1] * 10, self._walls.shape[0] * 10), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw walls
        for y in range(self._walls.shape[0]):
            for x in range(self._walls.shape[1]):
                if self._walls[y, x] == 1:
                    draw.rectangle([x * 10, y * 10, (x + 1) * 10, (y + 1) * 10], fill='black')

        # Draw startition and goal
        draw.ellipse([(start[1] * 10, start[0] * 10), (start[1] * 10 + 10, start[0] * 10 + 10)], fill='blue', width=0.25)
        draw.ellipse([(goal[1] * 10, goal[0] * 10), (goal[1] * 10 + 10, goal[0] * 10 + 10)], fill='red', width=0.25)

        # Draw trajectories
        for i in range(len(obs)):
            if terminals[i]:
                continue
            start = obs[i] - 0.5
            end = next_obs[i] - 0.5
            draw.line([(start[1] * 10 + 5, start[0] * 10 + 5), (end[1] * 10 + 5, end[0] * 10 + 5)], fill='red', width=1)
        
        # Draw grid
        for x in range(0, self._walls.shape[1] * 10, 10):
            draw.line([(x, 0), (x, self._walls.shape[0] * 10)], fill='gray', width=1)
        for y in range(0, self._walls.shape[0] * 10, 10):
            draw.line([(0, y), (self._walls.shape[1] * 10, y)], fill='gray', width=1)
        
        # Save the image
        if save_path is not None:
            img.save(save_path)

        img = img.resize((800, 800), Image.LANCZOS)

        # Convert PIL image to numpy array
        img_array = np.array(img)
        img_array = np.moveaxis(np.transpose(img_array), 0, -1)
        return img_array


    def get_env_frame_with_traj(self, start, goal, trajectories, save_path=None):
        trajectories = np.array(trajectories)
        start = start - 0.5
        goal = goal - 0.5

        # Create a blank image with a white background
        img = Image.new('RGB', (self._walls.shape[1] * 10, self._walls.shape[0] * 10), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw walls
        for y in range(self._walls.shape[0]):
            for x in range(self._walls.shape[1]):
                if self._walls[y, x] == 1:
                    draw.rectangle([x * 10, y * 10, (x + 1) * 10, (y + 1) * 10], fill='black')

        # Draw startition and goal
        draw.ellipse([(start[1] * 10, start[0] * 10), (start[1] * 10 + 10, start[0] * 10 + 10)], fill='blue', width=0.25)
        draw.ellipse([(goal[1] * 10, goal[0] * 10), (goal[1] * 10 + 10, goal[0] * 10 + 10)], fill='red', width=0.25)

        # Draw trajectories with fading colors
        for trajectory in trajectories:
            num_segments = len(trajectory) - 1
            for i in range(num_segments):
                start = trajectory[i] - 0.5
                end = trajectory[i + 1] - 0.5

                # Calculate the fading color (fade towards more transparent)
                opacity = int(255 * (1 - i / num_segments))
                color = (255, 0, 0, opacity)

                # Draw line segment
                draw.line([(start[1] * 10 + 5, start[0] * 10 + 5), (end[1] * 10 + 5, end[0] * 10 + 5)], fill=color, width=1)

        # Draw grid
        for x in range(0, self._walls.shape[1] * 10, 10):
            draw.line([(x, 0), (x, self._walls.shape[0] * 10)], fill='gray', width=1)
        for y in range(0, self._walls.shape[0] * 10, 10):
            draw.line([(0, y), (self._walls.shape[1] * 10, y)], fill='gray', width=1)

        # Save the image
        if save_path is not None:
            img.save(save_path)

        img = img.resize((800, 800), Image.LANCZOS)

        # Convert PIL image to numpy array
        img_array = np.array(img)
        img_array = np.moveaxis(np.transpose(img_array), 0, -1)
        return img_array

if __name__ == '__main__':
    pygame.init()

    # Set up the display
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('PointMassEnv Game')
    
    start_pos = [[12.5, 4.5]]
    goal = [[4.5, 12.5]]

    env = PointMassEnv(start=np.array(start_pos[0], dtype=np.float32), env_name="EmptyRoom",)
    env._goal = np.array(goal[-1], dtype=np.float32)
    obs, _ = env.reset()
    
    step_num = 0
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        img = env.render()

        # trajectories = [[[0,0],[1,1],[2,2]], [[3,3],[4,4],[5,5]]]
        # img = env.get_env_frame_with_traj(obs, env._goal, trajectories)

        # trajectories = np.load('dataset.npy', allow_pickle=True)

        # with open('dataset.npy', 'rb') as f:
        #     trajectories = pickle.load(f)

        # trajectories = trajectories['trajectories']
        # img = env.get_env_frame_with_traj(obs, env._goal, trajectories)

        img_surface = pygame.surfarray.make_surface(img)
        screen.blit(img_surface, (0, 0))
        pygame.display.flip()
        pygame.time.Clock().tick(30)

        action = [0, 0]
        # Key input handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = [0, -0.5]  # Move the image left
        if keys[pygame.K_RIGHT]:
            action = [0, 0.5]  # Move the image right
        if keys[pygame.K_UP]:
            action = [-0.5, 0]  # Move the image up
        if keys[pygame.K_DOWN]:
            action = [0.5, 0]  # Move the image down
        obs, reward, term, trunc, info = env.step(action)
        print('reward:', reward)
        print('step_num:', step_num)
        step_num+=1
        # print('term:', term)

    # Quit Pygame
    pygame.quit()
    sys.exit()