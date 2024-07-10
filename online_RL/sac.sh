#!/bin/bash
conda_env="corl_0"

project="discrete_FourRooms"
checkpoints_path_base="discrete_FourRooms"
env_names=("FourRooms")
reward_type="dense"
discrete_action="True"

seeds=(100)
GPUS=(1)

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
                                        python3 online_RL/sac.py \
                                        --env_name $env_name \
                                        --reward_type $reward_type \
                                        --discrete_action $discrete_action \
                                        --save_model True \
                                        --checkpoints_path $checkpoints_path \
                                        --wandb_project_name $project \
                                        --seed $seed" C-m

        # Increment the experiment counter
        experiment_counter=$((experiment_counter + 1))

        # Delay to avoid potential race conditions
        sleep 5
    done
done
