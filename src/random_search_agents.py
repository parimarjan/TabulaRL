from agent import *
import numpy as np

class BasicRandomSearch(Agent):

    def __init__(self, nState, nAction, epLen, scaling=0.0, alpha=0.1,
            batch_size=10, v=1.0, epsilon=0.1):

        # ignore this for now.
        # self.epLen = epLen

        self.alpha = alpha
        self.batch_size = batch_size
        self.v = v
        self.epsilon = epsilon

        self.theta = np.zeros((nState, nAction))
        self.cur_theta = np.zeros((nState, nAction))

    def _egreedy(self, Q):
        nAction = Q.size
        noise = np.random.rand()

        if noise < self.epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])

        return action

    def pick_action(self, state, timestep):
        qVals = self.cur_theta[state]
        return self._egreedy(qVals)
