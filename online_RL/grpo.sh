#!/bin/bash
conda_env="corl_0"

project="discrete_FourRooms_grpo"
checkpoints_path_base="discrete_FourRooms_grpo"
env_names=("FourRooms")
reward_type="sparse"
discrete_action="True"

seeds=(100)
GPUS=(1)

# (Optional) GRPO hyperparams you might want to sweep
group_size=8
groups_per_update=8
update_epochs=4
clip_coef=0.2
kl_beta=0.0
learning_rate=3e-4
total_timesteps=1000000
episode_length=200

# Initialize an experiment counter
experiment_counter=0

# Loop through each parameter set
for env_name in "${env_names[@]}"; do
  for seed in "${seeds[@]}"; do

    # Calculate the device number for the current session
    device_index=$(( experiment_counter % ${#GPUS[@]} ))
    device=${GPUS[$device_index]}

    # Construct the session name based on parameters
    session_name="${project}_env_name_${env_name}"
    session_name="${session_name}_seed_${seed}"
    session_name="${session_name}_G_${group_size}_GPU_${device}"
    session_name="${session_name//./_}" # Replace dots with underscores

    # Append session name to the checkpoints path
    checkpoints_path="${checkpoints_path_base}/${session_name}"
    checkpoints_path="${checkpoints_path//./_}" # Replace dots with underscores

    # Create a new tmux session with the session name
    tmux new-session -d -s $session_name

    # Activate the conda environment
    tmux send-keys -t $session_name "conda activate $conda_env" C-m

    # Start the experiment with the specified parameters
    tmux send-keys -t $session_name "CUDA_VISIBLE_DEVICES=$device \
      python3 online_RL/grpo.py \
      --env_name $env_name \
      --episode_length $episode_length \
      --reward_type $reward_type \
      --discrete_action $discrete_action \
      --save_model True \
      --checkpoints_path $checkpoints_path \
      --wandb_project_name $project \
      --seed $seed \
      --total_timesteps $total_timesteps \
      --group_size $group_size \
      --groups_per_update $groups_per_update \
      --update_epochs $update_epochs \
      --clip_coef $clip_coef \
      --kl_beta $kl_beta \
      --learning_rate $learning_rate" C-m

    # Increment the experiment counter
    experiment_counter=$((experiment_counter + 1))

    # Delay to avoid potential race conditions
    sleep 5
  done
done
