#!/bin/bash

# List of session names
env_1="antmaze-umaze-v2"
env_2="antmaze-umaze-v2"
project="toy_hyper"
conda_env="corl_0"
checkpoints_path_base="toy_hyper"

normalize_state=(False)
normalize_reward_values=(True False)
discount_values=(0.976 0.99)
true_dice_alpha_values=(0.5 1 100)
semi_dice_lambda_values=(0.3)

seed=(100)
GPUS=(0 1 2 3)

# Initialize an experiment counter
experiment_counter=0

# Loop through each parameter set
for normalize_reward in "${normalize_reward_values[@]}"; do
  for true_dice_alpha in "${true_dice_alpha_values[@]}"; do
    for discount in "${discount_values[@]}"; do
      for semi_dice_lambda in "${semi_dice_lambda_values[@]}"; do
        for current_seed in "${seed[@]}"; do
          # Calculate the device number for the current session
          device_index=$(( experiment_counter % ${#GPUS[@]} ))
          device=${GPUS[$device_index]}

          # Construct the session name based on parameters
          session_name="${project}_normalize_reward_${normalize_reward}_true_dice_alpha_${true_dice_alpha}_discount_${discount}_semi_dice_lambda_${semi_dice_lambda}_seed_${current_seed}"

          session_name="${session_name//./_}" # Replace dots with underscores

          # Append session name to the checkpoints path
          checkpoints_path="${checkpoints_path_base}/${session_name}"
          checkpoints_path="${checkpoints_path//./_}" # Replace dots with underscores

          # Create a new tmux session with the session name
          tmux new-session -d -s $session_name

          # Activate the conda environment
          tmux send-keys -t $session_name "source /data/shuozhe/miniconda3/bin/activate $conda_env" C-m
          # tmux send-keys -t $session_name "conda activate $conda_env" C-m

          # Start the experiment with the specified parameters
          tmux send-keys -t $session_name "CUDA_VISIBLE_DEVICES=$device \
                                          python3 vdice_correct_filter.py \
                                          --env_1 $env_1 \
                                          --env_2 $env_2 \
                                          --normalize_reward $normalize_reward \
                                          --true_dice_alpha $true_dice_alpha \
                                          --discount $discount \
                                          --semi_dice_lambda $semi_dice_lambda \
                                          --seed $current_seed \
                                          --max_timesteps 2000000 \
                                          --project $project \
                                          --checkpoints_path $checkpoints_path" C-m

          # Increment the experiment counter
          experiment_counter=$((experiment_counter + 1))

          # Delay to avoid potential race conditions
          sleep 5
        done
      done
    done
  done
done
