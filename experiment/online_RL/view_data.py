import os
import pickle
from collections import Counter, OrderedDict
import numpy as np
import sys

# 你的环境
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from PointMassEnv import PointMassEnv


def load_dataset(path: str):
    # 兼容你两种保存方式：pkl 或者 “用 pickle.dump 写出来的 .npy”
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def quantize_obs(obs: np.ndarray, decimals: int = 1):
    # 让 key 稳定：把浮点量化到网格（你收集时已经 round 到 0.1，这里再保险一次）
    return np.round(obs.astype(np.float32), decimals=decimals)


def quantize_act(act: np.ndarray, decimals: int = 3):
    # 离散动作一般是 -1/0/1；连续也可以 round 一下避免浮点抖动
    return np.round(act.astype(np.float32), decimals=decimals)


def compute_transition_weights_by_obs_act(obs, actions, obs_decimals=1, act_decimals=3):
    """
    返回：
      - weights: shape (N,) 每条 transition 的 “(obs, act) 出现次数”
      - counts: Counter 方便你debug
    """
    obs_q = quantize_obs(obs, obs_decimals)
    act_q = quantize_act(actions, act_decimals)

    keys = []
    for i in range(len(obs_q)):
        k = (float(obs_q[i, 0]), float(obs_q[i, 1]), float(act_q[i, 0]), float(act_q[i, 1]))
        keys.append(k)

    counts = Counter(keys)
    weights = np.asarray([counts[k] for k in keys], dtype=np.float32)
    return weights, counts


def dedup_transitions(obs, actions, next_obs, terminals, weights_obs_act,
                     obs_decimals=1, act_decimals=3, next_decimals=1):
    """
    可选：为了避免同一条边画很多遍导致“叠加变得过暗”，我们把 (obs, act, next_obs, terminal) 去重，只画一次。
    但透明度仍然用 (obs, act) 的频率 weights_obs_act。
    """
    obs_q = quantize_obs(obs, obs_decimals)
    act_q = quantize_act(actions, act_decimals)
    next_q = quantize_obs(next_obs, next_decimals)
    terminals = terminals.astype(bool)

    seen = OrderedDict()
    for i in range(len(obs)):
        k = (
            float(obs_q[i, 0]), float(obs_q[i, 1]),
            float(act_q[i, 0]), float(act_q[i, 1]),
            float(next_q[i, 0]), float(next_q[i, 1]),
            bool(terminals[i]),
        )
        if k not in seen:
            seen[k] = i

    idx = np.array(list(seen.values()), dtype=np.int64)

    return (
        obs[idx],
        actions[idx],
        next_obs[idx],
        terminals[idx],
        weights_obs_act[idx],
    )


def main():
    # ----------- 你需要改的路径 -----------
    dataset_path = "/media/shuozhe/Disk_Bottom_4TB/pointmas_grpo/08_bc_dataset_collected.pkl"   # 或者 hand_dataset.npy（你是 pickle.dump 出来的也能读）
    save_path = "env_frame_freq.png"
    # -------------------------------------

    data = load_dataset(dataset_path)

    obs = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    next_obs = np.asarray(data["next_observations"], dtype=np.float32)
    terminals = np.asarray(data["terminals"], dtype=np.float32)

    # (1) 算每条 transition 的 (obs, act) 出现次数
    weights, counts = compute_transition_weights_by_obs_act(obs, actions, obs_decimals=1, act_decimals=3)

    # 打印一下你举的例子：在 [2.5, 14.5] 做 [1, 0] 的次数
    example_key = (2.5, 14.5, 1.0, 0.0)
    print(f"count{example_key} = {counts.get(example_key, 0)}")

    # (2) 可选：去重（推荐打开，否则重复边画多遍会很黑）
    USE_DEDUP = True
    if USE_DEDUP:
        obs, actions, next_obs, terminals, weights = dedup_transitions(
            obs, actions, next_obs, terminals, weights,
            obs_decimals=1, act_decimals=3, next_decimals=1
        )
        print(f"After dedup: {len(obs)} transitions to draw")

    # (3) 构建 env（用你这次收集数据的地图/起点/终点）
    # 这里你按自己的 config 改就行
    env = PointMassEnv(
        env_name="GuidanceCorridorMaze",
        reward_type="sparse",
        start=np.array([2.5, 2.5], dtype=np.float32),
        goal=np.array([14.5, 2.5], dtype=np.float32),
        goal_radius=0.8,
        episode_length=120,
    )

    # (4) 调用你改好的 render：weights 就是出现次数
    img = env.get_env_frame_with_selected_traj_plt(
        start=env._start,
        goal=env._goal,
        obs=obs,
        next_obs=next_obs,
        terminals=terminals.astype(bool),
        actions=actions,
        transition_weights=weights,   # 核心：出现次数
        save_path=save_path,
        alpha_min=0.05,
        alpha_max=1.0,
        use_log=True,
    )

    print(f"Saved to: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
