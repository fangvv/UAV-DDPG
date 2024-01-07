# UAV-DDPG

This is the source code for our paper: **Computation Offloading Optimization for UAV-assisted Mobile Edge Computing: A Deep Deterministic Policy Gradient Approach**. A brief introduction of this work is as follows:

> Unmanned Aerial Vehicle (UAV) can play an important role in wireless systems as it can be deployed flexibly to help improve coverage and quality of communication. In this paper, we consider a UAV-assisted Mobile Edge Computing (MEC) system, in which a UAV equipped with computing resources can provide offloading services to nearby user equipments (UEs). The UE offloads a portion of the computing tasks to the UAV, while the remaining tasks are locally executed at this UE. Subject to constraints on discrete variables and energy consumption, we aim to minimize the maximum processing delay by jointly optimizing user scheduling, task offloading ratio, UAV flight angle and flight speed. Considering the non-convexity of this problem, the high-dimensional state space and the continuous action space, we propose a computation offloading algorithm based on Deep Deterministic Policy Gradient (DDPG) in Reinforcement Learning (RL). With this algorithm, we can obtain the optimal computation offloading policy in an uncontrollable dynamic environment. Extensive experiments have been conducted, and the results show that the proposed DDPG-based algorithm can quickly converge to the optimum. Meanwhile, our algorithm can achieve a significant improvement in processing delay as compared with baseline algorithms, e.g., Deep Q Network (DQN).

This work will be published by Wireless Networks. Click [here](https://link.springer.com/article/10.1007/s11276-021-02632-z) for our paper online.

## Required software

TensorFlow 1.X

## Citation

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

We have another work on [MADDPG](https://github.com/fangvv/VN-MADDPG) for your reference, and you can simply use Ray for implementing [DRL algorithms](https://github.com/ray-project/ray/tree/master/rllib/algorithms) now.

## Contact

Yunpeng Wang (1609791621@qq.com)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.