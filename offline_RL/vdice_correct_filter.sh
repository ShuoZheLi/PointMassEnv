#!/bin/bash

# List of session names
env_1="antmaze-umaze-v2"
env_2="antmaze-umaze-v2"
conda_env="corl_0"
project="2_mature_v_q_goal_1_wall_n10"
checkpoints_path_base="2_mature_v_q_goal_1_wall_n10"

env_name="EmptyRoom"
discrete_action="True"
normalize_state=(False)
normalize_reward_values=(True)
discount_values=(0.99)
# semi_dice_lambda_values=(0.2 0.4 0.6 0.8)
# true_dice_alpha_values=(1)
# semi_dice_lambda_values=(0.2)
# true_dice_alpha_values=(1.5 2 3 4)

# semi_dice_lambda_values=(0.7)
# true_dice_alpha_values=(0.5)

semi_dice_lambda_values=(0.5)
true_dice_alpha_values=(0.25)

percent_expert="0"

eval_freq="5000"
save_freq="5000"

batch_size="256"
hidden_dim="256"


seed=(19990526)
# GPUS=(2 3 1)
GPUS=(3)

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
          # append percent_expert
          session_name="${session_name}_percent_expert_${percent_expert}"
          # append eval_freq
          session_name="${session_name}_eval_freq_${eval_freq}"
          # append save_freq
          session_name="${session_name}_save_freq_${save_freq}"
          # append batch_size
          session_name="${session_name}_batch_size_${batch_size}"
          # append hidden_dim
          session_name="${session_name}_hidden_dim_${hidden_dim}"

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
                                          python3 offline_RL/vdice_correct_filter.py \
                                          --env_name $env_name \
                                          --discrete_action $discrete_action \
                                          --percent_expert $percent_expert \
                                          --eval_freq $eval_freq \
                                          --save_freq $save_freq \
                                          --batch_size $batch_size \
                                          --hidden_dim $hidden_dim \
                                          --env_1 $env_1 \
                                          --env_2 $env_2 \
                                          --normalize_reward $normalize_reward \
                                          --true_dice_alpha $true_dice_alpha \
                                          --discount $discount \
                                          --semi_dice_lambda $semi_dice_lambda \
                                          --seed $current_seed \
                                          --max_timesteps 1000000 \
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
