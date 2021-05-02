import numpy as np
from UAV_env import UAVEnv

env = UAVEnv()
M = env.M


class StateNormalization(object):
    def __init__(self):
        self.high_state = np.array(
            [5e5, env.ground_length, env.ground_width, 100 * 1048576])
        self.high_state = np.append(self.high_state, np.ones(M * 2) * env.ground_length)
        self.high_state = np.append(self.high_state, np.ones(M) * 3145728)
        self.high_state = np.append(self.high_state, np.ones(M))
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        # self.high_state = np.array(
        #     [500000, 100, 100, 100 * 1048576, 100, 100, 100, 100, 100, 100, 100, 100, 2097152, 2097152, 2097152, 2097152,
        #      1, 1, 1, 1])  # uav loc, ue loc, task size, block_flag
        self.low_state = np.zeros(20)  # uav loc, ue loc, task size, block_flag
        self.low_state[len(self.low_state) - 2 * M:len(self.low_state) - M] = np.ones(M) * 2621440

    def state_normal(self, state):
        state[len(state) - 2 * M: len(state) - M] -= 2621440
        res = state / (self.high_state - self.low_state)
        # res = np.round(res, 2)
        return res
