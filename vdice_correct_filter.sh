#!/bin/bash

# List of session names
env_1="halfcheetah-expert-v2"
env_2="halfcheetah-random-v2"
project="toy_example"
conda_env="corl_0"
checkpoints_path_base="toy_example"

expert_num=(10000)
# expert_num=(100000)
bc_max_timesteps=(600000)

# semi_vdice_lambda=(0.5 0.6 0.7 0.8 0.9)
semi_vdice_lambda=(0.525)
true_vdice_lambda=(0.99)

semi_lambda_delta=(0)
true_lambda_delta=(0)

seed=(100)

update_freq=(100000)

layernorm="False"

policy_ratio="False"
state_ratio="False"
or_ration="True"
and_ratio="False"

ed_name="or_ration"

# specify GPUs
# GPUS=(0 1 2 3)
GPUS=(1)

# Initialize an experiment counter
experiment_counter=0

# Loop through each parameter set
for current_seed in "${seed[@]}"; do
  for current_semi_vdice_lambda in "${semi_vdice_lambda[@]}"; do
    for current_true_vdice_lambda in "${true_vdice_lambda[@]}"; do
      for current_true_lambda_delta in "${true_lambda_delta[@]}"; do
        for current_semi_lambda_delta in "${semi_lambda_delta[@]}"; do
          for current_update_freq in "${update_freq[@]}"; do
            for current_expert_num in "${expert_num[@]}"; do
              for current_bc_max_timesteps in "${bc_max_timesteps[@]}"; do
                # Calculate the device number for the current session
                device_index=$(( experiment_counter % ${#GPUS[@]} ))
                device=${GPUS[$device_index]}

                # Construct the session name based on parameters and ed_name
                session_name="${project}_${current_true_lambda_delta}_${current_semi_lambda_delta}_semi_lambda_${current_semi_vdice_lambda}_true_lambda_${current_true_vdice_lambda}_seed_${current_seed}_update_freq_${current_update_freq}_expert_num_${current_expert_num}_current_bc_max_timesteps_${current_bc_max_timesteps}_layernorm_${layernorm}_${ed_name}"

                session_name="${session_name//./_}" # Replace dots with underscores

                # Append session name to the checkpoints path
                checkpoints_path="${checkpoints_path_base}/${session_name}"
                checkpoints_path="${checkpoints_path//./_}" # Replace dots with underscores

                # Create a new tmux session with the session name
                tmux new-session -d -s $session_name

                # Activate the conda environment
                tmux send-keys -t $session_name "source /data/shuozhe/miniconda3/bin/activate $conda_env" C-m

                # Start the experiment with the specified parameters
                tmux send-keys -t $session_name "CUDA_VISIBLE_DEVICES=$device \
                                                python3 vdice_correct_filter.py \
                                                --env_1 $env_1 \
                                                --env_2 $env_2 \
                                                --true_lambda_delta $current_true_lambda_delta \
                                                --semi_lambda_delta $current_semi_lambda_delta \
                                                --semi_vdice_lambda $current_semi_vdice_lambda \
                                                --true_vdice_lambda $current_true_vdice_lambda \
                                                --update_freq $current_update_freq \
                                                --seed $current_seed \
                                                --max_timesteps 2000000 \
                                                --project $project \
                                                --checkpoints_path $checkpoints_path \
                                                --policy_ratio $policy_ratio \
                                                --state_ratio $state_ratio \
                                                --or_ration $or_ration \
                                                --and_ratio $and_ratio \
                                                --expert_num $current_expert_num \
                                                --bc_max_timesteps $current_bc_max_timesteps \
                                                --layernorm $layernorm" C-m

                # Increment the experiment counter
                experiment_counter=$((experiment_counter + 1))

                # Delay to avoid potential race conditions
                sleep 5
              done
            done
          done
        done
      done
    done
  done
done