# UAV-DDPG

This is the source code for our paper: **Computation Offloading Optimization for UAV-assisted Mobile Edge Computing: A Deep Deterministic Policy Gradient Approach**. A brief introduction of this work is as follows:

> Unmanned Aerial Vehicle (UAV) can play an important role in wireless systems as it can be deployed flexibly to help improve coverage and quality of communication. In this paper, we consider a UAV-assisted Mobile Edge Computing (MEC) system, in which a UAV equipped with computing resources can provide offloading services to nearby user equipments (UEs). The UE offloads a portion of the computing tasks to the UAV, while the remaining tasks are locally executed at this UE. Subject to constraints on discrete variables and energy consumption, we aim to minimize the maximum processing delay by jointly optimizing user scheduling, task offloading ratio, UAV flight angle and flight speed. Considering the non-convexity of this problem, the high-dimensional state space and the continuous action space, we propose a computation offloading algorithm based on Deep Deterministic Policy Gradient (DDPG) in Reinforcement Learning (RL). With this algorithm, we can obtain the optimal computation offloading policy in an uncontrollable dynamic environment. Extensive experiments have been conducted, and the results show that the proposed DDPG-based algorithm can quickly converge to the optimum. Meanwhile, our algorithm can achieve a significant improvement in processing delay as compared with baseline algorithms, e.g., Deep Q Network (DQN).

> 无人机（UAV）因其可灵活部署以改善通信覆盖范围和质量的能力，在无线系统中扮演着重要角色。本文研究一种无人机辅助的移动边缘计算（MEC）系统，其中配备计算资源的无人机可为附近用户设备（UE）提供任务卸载服务。用户设备将部分计算任务卸载至无人机，其余任务则在本地执行。在离散变量约束和能耗限制条件下，我们通过联合优化用户调度、任务卸载比例、无人机飞行角度与飞行速度，以最小化最大处理延迟。针对该问题的非凸特性、高维状态空间及连续动作空间，我们提出了一种基于强化学习（RL）中深度确定性策略梯度（DDPG）的计算卸载算法。通过该算法，我们可以在不可控的动态环境中获得最优计算卸载策略。大量实验结果表明，所提出的基于DDPG的算法能快速收敛至最优解，同时相较于深度Q网络（DQN）等基线算法，我们的算法在处理延迟方面实现了显著提升。

This work will be published by Wireless Networks. Click [here](https://link.springer.com/article/10.1007/s11276-021-02632-z) for our paper online.

## Required software

- TensorFlow 1.X
- NumPy
- Matplotlib

## Project Structure

```
UAV-DDPG/
├── DDPG/                              # DDPG algorithm (main algorithm)
│   ├── UAV_env.py                    # Environment simulator: UAV, UE, channel model, energy model
│   ├── ddpg_algo.py                  # DDPG agent: Actor-Critic networks, experience replay, training loop
│   ├── state_normalization.py        # State normalization: scale state values to [0, 1]
│   ├── DDPG_without_behavior_noise/  # Ablation: DDPG without exploration noise
│   └── DDPG_without_state_normalization/  # Ablation: DDPG without state normalization
├── DQN/                               # DQN baseline algorithm
│   ├── UAV_env.py                    # Environment with discrete action space (for DQN)
│   ├── dqn_algo.py                   # DQN algorithm implementation
│   └── state_normalization.py
├── Actor Critc/                       # Actor-Critic baseline algorithm
│   ├── UAV_env.py
│   ├── ac_algo.py                    # Actor-Critic with continuous action space
│   └── state_normalization.py
├── Edge_only/                         # Edge-only baseline (offload all tasks to UAV)
├── Local_only/                        # Local-only baseline (no offloading)
└── README.md
```

## Core Modules

### UAVEnv (`UAV_env.py`)

The environment class that models the UAV-assisted MEC system. Key attributes:

| Attribute | Description |
|-----------|-------------|
| `height = ground_length = ground_width = 100` | 3D area: 100m × 100m × 100m |
| `B = 1 MHz` | Channel bandwidth |
| `flight_speed = 50 m/s` | UAV flight speed |
| `f_ue` / `f_uav` | CPU frequency of UE / UAV |
| `M = 4` | Number of user equipments |
| `slot_num = T / (t_fly + t_com)` | 40 time slots per episode |

**State space** (`state_dim = 4 + M × 4`):

| Index | Feature | Dimension |
|-------|---------|-----------|
| 0 | UAV battery remaining | 1 |
| 1–2 | UAV location (x, y) | 2 |
| 3 | Remaining total task size | 1 |
| 4–(3+M×2) | UE locations (x, y) for each UE | M × 2 |
| (4+M×2)–(3+M×3) | Task size of each UE | M |
| (4+M×3)–(3+M×4) | Block flag (LOS/NLOS) of each UE | M |

**Action space** (`action_dim = 4`): continuous values in `[-1, 1]` mapped to `[0, 1]` for:

| Index | Meaning | Range |
|-------|---------|-------|
| 0 | Target UE index | `[0, M-1]` |
| 1 | Flight angle θ | `[0, 2π]` |
| 2 | Flight distance | `[0, flight_speed × t_fly]` |
| 3 | Task offloading ratio | `[0, 1]` |

**Key methods:**

- `reset()` — Reset the environment to the initial state and return the observation.
- `step(action)` — Execute one MDP step: decode the action, compute flight energy, update UAV position, calculate delay via `com_delay()`, update battery/task/UE states, and return `(next_state, reward, terminal, step_redo, ...)`.
- `com_delay(loc_ue, loc_uav, offloading_ratio, task_size, block_flag)` — Compute the maximum processing delay as `max(transmission + edge computation, local computation)`. The channel gain follows free-space path loss with LOS/NLOS noise levels.

**Reward:** `-delay`, i.e., the negative of the maximum processing delay. The algorithm minimizes the maximum delay across all UEs.

### DDPG (`ddpg_algo.py`)

The DDPG agent with experience replay and soft target updates.

**Network architecture:**

| Network | Layers | Output Activation |
|---------|--------|-------------------|
| Actor | 400 → 300 → 10 → 4 | tanh, scaled by `action_bound` |
| Critic | state(400) + action(400) → 300 → 10 → 1 | linear (Q-value) |

**Key hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LR_A = 0.001` | Learning rate for Actor |
| `LR_C = 0.002` | Learning rate for Critic |
| `GAMMA = 0.001` | Discount factor |
| `TAU = 0.01` | Soft target update rate |
| `MEMORY_CAPACITY = 10000` | Experience replay buffer size |
| `BATCH_SIZE = 64` | Training batch size |

**Key methods:**

- `choose_action(s)` — Forward pass through the Actor network to select a deterministic action.
- `store_transition(s, a, r, s_)` — Store a transition tuple in the replay buffer.
- `learn()` — Soft-update target networks, then sample a batch and train both Actor (maximize Q) and Critic (minimize TD error).
- `_build_a(s, scope, trainable)` — Build the Actor network: `state → 400(relu6) → 300(relu6) → 10(relu) → action(tanh)`.
- `_build_c(s, a, scope, trainable)` — Build the Critic network: `state` and `action` are separately embedded into 400 units, summed with bias, then `→ 300(relu6) → 10(relu) → Q-value`.

**Training loop:** For each episode, reset the environment, then for each step add Gaussian exploration noise `N(0, var)` to the action, execute `step()`, store the transition, and call `learn()` once the replay buffer has enough samples.

### StateNormalization (`state_normalization.py`)

Normalizes the raw state vector to `[0, 1]` by dividing each component by its maximum possible value. This accelerates training convergence and stabilizes the neural network.

| State component | Max value |
|-----------------|-----------|
| UAV battery | 500000 J |
| UAV location | 100 m |
| Remaining task size | 100 × 1048576 bits |
| UE location | 100 m |
| UE task size | ~2.5 Mbits |
| Block flag | 1 |

### DQN Baseline (`dqn_algo.py`)

The DQN baseline discretizes the continuous action space into `M × 11³` discrete actions for comparison. It uses a dual-network architecture (eval/target) with hard replacement every 200 steps.

### Ablation Studies

Two variants under `DDPG/` verify the contribution of key components:

- **DDPG_without_behavior_noise** — Removes the Gaussian exploration noise during training.
- **DDPG_without_state_normalization** — Feeds raw (unnormalized) states to the networks.

## Usage

```bash
# Install dependencies (TensorFlow 1.x required)
pip install tensorflow==1.14.0 numpy matplotlib

# Run DDPG (main algorithm)
cd DDPG
python ddpg_algo.py

# Run DQN baseline
cd DQN
python dqn_algo.py

# Run Actor-Critic baseline
cd "Actor Critc"
python ac_algo.py
```

## Citation

If you find UAV-DDPG useful or relevant to your project and research, please kindly cite our paper:

	@article{wang2021computation,
  		title={Computation offloading optimization for UAV-assisted mobile edge computing: a deep deterministic policy gradient approach},
  		author={Wang, Yunpeng and Fang, Weiwei and Ding, Yi and Xiong, Naixue},
  		journal={Wireless Networks},
  		volume={27},
  		number={4},
  		pages={2991--3006},
  		year={2021},
  		publisher={Springer}
	}
	
## Stargazers over time

[![Stargazers over time](https://starchart.cc/fangvv/UAV-DDPG.svg)](https://starchart.cc/fangvv/UAV-DDPG)

## For more

We have another work on [MADDPG](https://github.com/fangvv/VN-MADDPG) and [IMPALA](https://github.com/fangvv/MTACP) for your reference, and you can simply use Ray for implementing [DRL algorithms](https://github.com/ray-project/ray/tree/master/rllib/algorithms) now.

## Contact

Yunpeng Wang (1609791621@qq.com)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.