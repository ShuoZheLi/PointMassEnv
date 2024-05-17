import numpy as np
import matplotlib.pyplot as plt

# Define the array
grid = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Create the plot
plt.figure(figsize=(8, 8))
extent = [0, grid.shape[1], 0, grid.shape[0]]
plt.imshow(grid, cmap='Greys', interpolation='none', extent=extent)




old_traj = np.load('traj.npy', allow_pickle=True)
# assign each trajectory an id, count from 0
traj = old_traj.copy()


id = 0

new_traj = []
# assign each trajectory an id, count from 0
for i in range(traj.shape[0]):
    new_traj.append([])
    for j in range(traj[i].shape[0]):
        new_traj[i].append(np.append(traj[i][j], id))
        id += 1
    new_traj[i] = np.array(new_traj[i])

traj = np.array(new_traj)


# load in an array of id
load_id = np.load('/media/shuozhe/Disk_Bottom_4TB/corl/toycase/test/100_0st-hopper-expert-v2-776a6697_state_ratio_id/checkpoint_522999.npy', allow_pickle=True)


# use the id to filter out the trajectory, make sure the code is correct and logical, please write how the code works
for i in range(traj.shape[0]):
    traj[i] = traj[i][np.isin(traj[i][:,2], load_id)]



# traj = old_traj[-10:]
traj = traj[-10:]


for i in range(traj.shape[0]):
    num_points = traj[i].shape[0]
    traj_init = traj[i]
    for i in range(num_points - 1):
        alpha = (num_points - i) / num_points
        plt.plot(traj_init[i:i+2, 1], traj_init[i:i+2, 0], color='purple', alpha=alpha)


plt.grid(color='gray', linestyle='-', linewidth=0.25)

# change the size of the grid box to be 1x1
plt.xticks(np.arange(0, 19, 1))
plt.yticks(np.arange(0, 19, 1))
plt.grid(True)



# plt.axis('off')
plt.title('Four-room')
plt.gca().invert_yaxis()
plt.show()
