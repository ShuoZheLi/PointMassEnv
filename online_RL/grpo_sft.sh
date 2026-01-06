#!/bin/bash
set -euo pipefail

conda_env="corl_0"

# -----------------------------
# Project / run naming
# -----------------------------
project="discrete_GuidanceCorridorMaze_GRPO_SFT"
checkpoints_path_base="discrete_GuidanceCorridorMaze_GRPO_SFT"

env_names=("GuidanceCorridorMaze")
reward_type="sparse"
# reward_type="dense"
discrete_action="True"

# Start / goal (now supported by the updated grpo_sft.py)
start="2.5,14.5"
goal="14.5,2.5"
goal_radius=0.8

seeds=(100)
GPUS=(0)

# -----------------------------
# GRPO hyperparams
# -----------------------------
group_size=8
groups_per_update=8
update_epochs=4
clip_coef=0.2
kl_beta=0.0
learning_rate=3e-4
total_timesteps=1000000
episode_length=200

# -----------------------------
# SFT / BC term params
# -----------------------------
sft_weight=0.8
sft_weight_end=0.0
sft_anneal_updates=100
sft_dataset_path="hand_dataset.npy"
sft_minibatch_size=64

# Initialize an experiment counter
experiment_counter=0

for env_name in "${env_names[@]}"; do
  for seed in "${seeds[@]}"; do

    device_index=$(( experiment_counter % ${#GPUS[@]} ))
    device=${GPUS[$device_index]}

    # Build tmux session name
    session_name="${project}_env_${env_name}_reward_type_${reward_type}_seed_${seed}"
    session_name="${session_name}_G_${group_size}_GPU_${device}"
    session_name="${session_name}_sftW_${sft_weight}"
    session_name="${session_name}_st_${start}_gl_${goal}"
    session_name="${session_name//./_}"   # replace dots
    session_name="${session_name//,/__}"  # replace commas (tmux-friendly)
    session_name="${session_name// /}"    # remove spaces

    # Checkpoints path mirrors session name
    checkpoints_path="${checkpoints_path_base}/${session_name}"
    checkpoints_path="${checkpoints_path//./_}"
    checkpoints_path="${checkpoints_path//,/__}"
    checkpoints_path="${checkpoints_path// /}"

    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "conda activate $conda_env" C-m

    # Launch training
    tmux send-keys -t "$session_name" "CUDA_VISIBLE_DEVICES=$device \
python3 online_RL/grpo_sft.py \
  --env_name $env_name \
  --reward_type $reward_type \
  --discrete_action $discrete_action \
  --episode_length $episode_length \
  --start \"$start\" \
  --goal \"$goal\" \
  --goal_radius $goal_radius \
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
  --learning_rate $learning_rate \
  --sft_weight $sft_weight \
  --sft_weight_end $sft_weight_end \
  --sft_anneal_updates $sft_anneal_updates \
  --sft_dataset_path $sft_dataset_path \
  --sft_minibatch_size $sft_minibatch_size" C-m

    experiment_counter=$((experiment_counter + 1))
    sleep 5
  done
done
