#!/bin/bash

# List of session names
env_1="antmaze-umaze-v2"
env_2="antmaze-umaze-v2"
conda_env="corl_0"
project="midwall_reproduce"
checkpoints_path_base="midwall_reproduce"

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

semi_dice_lambda_values=(0.55 0.575 0.6 0.625)
# semi_dice_lambda_values=(0.6 0.625 0.65 0.675)
# semi_dice_lambda_values=(0.5 0.525 0.55 0.575)
true_dice_alpha_values=(1.25)
semi_q_alpha_values=(0.5 0.75 1.0 1.25)

percent_expert="0"

eval_freq="15000"
save_freq="15000"

# batch_size_values=(64 128 256)
batch_size_values=(64)
hidden_dim_values=(256)


seed=(19990526)
GPUS=(0 1 2 3)
# GPUS=(1 2 3)

# Initialize an experiment counter
experiment_counter=0

# Loop through each parameter set
for normalize_reward in "${normalize_reward_values[@]}"; do
  for hidden_dim in "${hidden_dim_values[@]}"; do
    for batch_size in "${batch_size_values[@]}"; do
      for true_dice_alpha in "${true_dice_alpha_values[@]}"; do
        for discount in "${discount_values[@]}"; do
          for semi_dice_lambda in "${semi_dice_lambda_values[@]}"; do
            for semi_q_alpha in "${semi_q_alpha_values[@]}"; do
              for current_seed in "${seed[@]}"; do
                # Calculate the device number for the current session
                device_index=$(( experiment_counter % ${#GPUS[@]} ))
                device=${GPUS[$device_index]}

                # Construct the session name based on parameters
                session_name="${project}"
                # append discount
                # session_name="${session_name}_discount_${discount}"
                # # append normalize_reward
                # session_name="${session_name}_normalize_reward_${normalize_reward}"
                # # append true_dice_alpha
                # session_name="${session_name}_true_dice_alpha_${true_dice_alpha}"
                # append semi_dice_lambda
                session_name="${session_name}_semi_dice_lambda_${semi_dice_lambda}"
                # # append seed
                # session_name="${session_name}_seed_${current_seed}"
                # # append percent_expert
                # session_name="${session_name}_percent_expert_${percent_expert}"
                # # append eval_freq
                # session_name="${session_name}_eval_freq_${eval_freq}"
                # # append save_freq
                # session_name="${session_name}_save_freq_${save_freq}"
                # append batch_size
                # session_name="${session_name}_batch_size_${batch_size}"
                # # append hidden_dim
                # session_name="${session_name}_hidden_dim_${hidden_dim}"
                # append semi_q_alpha
                session_name="${session_name}_semi_q_alpha_${semi_q_alpha}"

                session_name="${session_name//./_}" # Replace dots with underscores

                # alg="semi_q_alpha_${semi_q_alpha}"
                alg=" "
                alg="${session_name}"
                # alg="${alg}_hidden_dim_${hidden_dim}"
                # alg="${alg}_batch_size_${batch_size}"
                

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
                                                python3 offline_RL/vdice_correct_filter_paper_ds_method.py \
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
                                                --semi_q_alpha $semi_q_alpha \
                                                --seed $current_seed \
                                                --max_timesteps 1000000 \
                                                --project $project \
                                                --checkpoints_path $checkpoints_path \
                                                --alg $alg" C-m

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