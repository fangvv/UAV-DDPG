import numpy as np
from UAV_env import UAVEnv


class StateNormalization(object):
    def __init__(self):
        env = UAVEnv()
        M = env.M
        self.high_state = np.array(
            [5e5, env.ground_length, env.ground_width, 100 * 1048576])
        self.high_state = np.append(self.high_state, np.ones(M * 2) * env.ground_length)
        self.high_state = np.append(self.high_state, np.ones(M) * 2621440)
        self.high_state = np.append(self.high_state, np.ones(M))

        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.high_state = np.array(
        #     [500000, 100, 100, 60 * 1048576, 100, 100, 100, 100, 100, 100, 100, 100, 2097152, 2097152, 2097152, 2097152,
        #      1, 1, 1, 1])  # uav loc, ue loc, task size, block_flag
        self.low_state = np.zeros(4 * M + 4)  # uav loc, ue loc, task size, block_flag

    def state_normal(self, state):
        return state / (self.high_state - self.low_state)
