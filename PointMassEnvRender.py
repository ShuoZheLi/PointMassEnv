from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

##########################
##### render methods #####
##########################
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

def get_env_frame_with_selected_traj_plt(self, start=None, goal=None, obs=None, next_obs=None, trajectories=None, terminals=None, save_path=None):
    
    if start is None:
        start = self._start
        
    if goal is None:
        goal = self._goal
    start = start - 0.5
    goal = goal - 0.5

    fig, ax = plt.subplots()

    # Draw walls
    for y in range(self._walls.shape[0]):
        for x in range(self._walls.shape[1]):
            if self._walls[y, x] == 1:
                rect = patches.Rectangle((x, y), 1, 1, linewidth=0, edgecolor='none', facecolor='black')
                ax.add_patch(rect)

    # Draw start and goal
    ax.add_patch(patches.Circle((start[1] + 0.5, start[0] + 0.5), 0.5, color='blue'))
    ax.add_patch(patches.Circle((goal[1] + 0.5, goal[0] + 0.5), 0.5, color='red'))

    # Draw trajectories
    for i in range(len(obs)):
        if terminals[i]:
            continue
        start = obs[i] - 0.5
        end = next_obs[i] - 0.5
        ax.plot([start[1] + 0.5, end[1] + 0.5], [start[0] + 0.5, end[0] + 0.5], color='red', linewidth=1)

    if trajectories is not None:
        for trajectory in trajectories:
            traj_points = np.array(trajectory) - 0.5
            ax.plot(traj_points[:, 1] + 0.5, traj_points[:, 0] + 0.5, color='blue', linewidth=1)

    # Draw grid
    ax.set_xticks(np.arange(0, self._walls.shape[1], 1))
    ax.set_yticks(np.arange(0, self._walls.shape[0], 1))
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, self._walls.shape[1])
    ax.set_ylim(0, self._walls.shape[0])
    ax.set_aspect('equal')

    # TODO: test this
    canvas = fig.canvas
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img_array = np.moveaxis(np.transpose(img_array), 0, -1)

    # Save the image
    if save_path is not None:
        img_array.save(save_path)
    
    return img_array

def get_env_frame_with_selected_traj(self, start=None, goal=None, obs=None, next_obs=None, trajectories=None, terminals=None, save_path=None):
    
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
    
    if trajectories is not None:
        self.draw_trajectory(draw, trajectories, color='blue')
    
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

def draw_arrowhead(self, draw, start, end, arrow_size=2.5, color='green'):
    angle = np.arctan2(end[0] - start[0], end[1] - start[1])
    arrow_p1 = (end[1] * 10 + 5 - arrow_size * np.cos(angle - np.pi / 6),
                end[0] * 10 + 5 - arrow_size * np.sin(angle - np.pi / 6))
    arrow_p2 = (end[1] * 10 + 5 - arrow_size * np.cos(angle + np.pi / 6),
                end[0] * 10 + 5 - arrow_size * np.sin(angle + np.pi / 6))
    draw.polygon([arrow_p1, (end[1] * 10 + 5, end[0] * 10 + 5), arrow_p2], fill=color)

def draw_trajectory(self, draw, trajectories, color='red'):
    # Draw trajectories with fading colors
    for trajectory in trajectories:
        num_segments = len(trajectory) - 1
        for i in range(num_segments):
            start = trajectory[i] - 0.5
            end = trajectory[i + 1] - 0.5

            # Draw line segment
            draw.line([(start[1] * 10 + 5, start[0] * 10 + 5), (end[1] * 10 + 5, end[0] * 10 + 5)], fill=color, width=1)
            self.draw_arrowhead(draw, start, end)

def get_env_frame_with_traj(self, start, goal, trajectories=None, save_path=None):
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
    
    self.draw_trajectory(draw, trajectories)

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