#!/bin/bash
set -euo pipefail

conda_env="corl_0"

# -----------------------------
# Project / run naming
# -----------------------------
project="mislead_BC"
checkpoints_path_base="mislead_BC"

env_names=("GuidanceCorridorMaze")
reward_type="sparse"

discretize_eval="True"

# Start / goal
start="2.5,2.5"
goal="14.5,2.5"
goal_radius=0.8

seeds=(100)
GPUS=(0)

# -----------------------------
# BC hyperparams (bc_new.py)
# -----------------------------
epochs=20000
batch_size=64
lr=3e-4
weight_decay=0.0
hidden=256

# IMPORTANT: bc_new.py expects these names
episode_length=200
max_episode_steps=200   # can be 512 if you want a bigger safety cap

dataset_path="mislead_hand_dataset.npy"
val_ratio=0.0

# -----------------------------
# Saving / eval knobs (bc_new.py)
# -----------------------------
save_model="True"
save_every_epochs=50
eval_every_epochs=50
eval_episodes=10

terminate_on_wall="False"

experiment_counter=0

for env_name in "${env_names[@]}"; do
  for seed in "${seeds[@]}"; do

    device_index=$(( experiment_counter % ${#GPUS[@]} ))
    device=${GPUS[$device_index]}

    session_name="${project}_env_${env_name}_reward_type_${reward_type}_seed_${seed}"
    session_name="${session_name}_GPU_${device}"
    session_name="${session_name}_bs_${batch_size}_lr_${lr}_ep_${epochs}"
    session_name="${session_name}_st_${start}_gl_${goal}"
    session_name="${session_name//./_}"
    session_name="${session_name//,/__}"
    session_name="${session_name// /}"

    checkpoints_path="${checkpoints_path_base}/${session_name}"
    checkpoints_path="${checkpoints_path//./_}"
    checkpoints_path="${checkpoints_path//,/__}"
    checkpoints_path="${checkpoints_path// /}"

    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "conda activate $conda_env" C-m

    cmd=(
      python3 offline_RL/bc_new.py
      --env_name "$env_name"
      --reward_type "$reward_type"
      --start "$start"
      --goal "$goal"
      --goal_radius "$goal_radius"

      --checkpoints_path "$checkpoints_path"
      --wandb_project_name "$project"
      --seed "$seed"

      --dataset_path "$dataset_path"
      --val_ratio "$val_ratio"

      --epochs "$epochs"
      --batch_size "$batch_size"
      --lr "$lr"
      --weight_decay "$weight_decay"
      --hidden "$hidden"

      --eval_every_epochs "$eval_every_epochs"
      --eval_episodes "$eval_episodes"

      --episode_length "$episode_length"
      --max_episode_steps "$max_episode_steps"

      --save_model "$save_model"
      --save_every_epochs "$save_every_epochs"

      --terminate_on_wall "$terminate_on_wall"
      --discretize_eval "$discretize_eval"
    )

    # DO NOT use printf %q (it injects backslashes into commas)
    cmd_str="${cmd[*]}"
    tmux send-keys -t "$session_name" "CUDA_VISIBLE_DEVICES=$device $cmd_str" C-m

    experiment_counter=$((experiment_counter + 1))
    sleep 2
  done
done
