import numpy as np


class StateNormalization(object):
    def __init__(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.high_state = np.array(
            [500000, 100, 100, 60 * 1048576, 100, 100, 100, 100, 100, 100, 100, 100, 2097152, 2097152, 2097152, 2097152,
             1, 1, 1, 1])  # uav loc, ue loc, task size, block_flag
        self.low_state = np.zeros(20)  # uav loc, ue loc, task size, block_flag

    def state_normal(self, state):
        return state / (self.high_state - self.low_state)
