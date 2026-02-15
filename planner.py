import numpy as np
from math import *

EW = 1.4
EU = 1.4


class FormationPlanner:
    def __init__(self, cluster_size=0):
        self.n_agent = cluster_size

    def update_states(self, state, id=-1):
        if id == -1:
            self.states = state.copy()
        else:
            self.states[id, :] = state.copy()

        self.plan()

    def plan():
        raise NotImplementedError

    def get_desired_state(self, id=0):
        return self.desired_states[id, :]


class CooperativeLocalizer:
    def __init__(self):
        self.t = 0
        self.state = None
        self.control = None
        self.observations = None

    def set_state(self, t, v, control, observations, estimates):
        self.control = control
        self.t = t
        self.observations = observations
        v1_hat = sqrt(estimates[1] ** 2 + estimates[3] ** 2)
        phi_hat = atan2(estimates[3], estimates[1])
        self.state = [v, v1_hat, phi_hat]

    def get_error_bound(self, t, observations):
        d_k = self.observations[0]
        theta_k = self.observations[1]

        v = self.state[0]
        v1_hat = self.state[1]
        phi_hat = self.state[2]
        w = self.control[1]

        del_t = t - self.t

        d_k1_hat = d_k
        d_k1_hat += (v1_hat * cos(theta_k - phi_hat) - v * cos(theta_k)) * del_t
        theta_k1_hat = theta_k
        theta_k1_hat += (
            (v * sin(theta_k) - v1_hat * sin(theta_k - phi_hat)) * del_t / d_k
        )
        theta_k1_hat -= w * del_t

        d_k1 = observations[0]
        theta_k1 = observations[1]

        e_u = (d_k1_hat - d_k1) / del_t
        e_w = d_k * (theta_k1 - theta_k1_hat) / del_t

        # if abs(d_k - d_k1) <= 0:
            # print(self.observations, observations)
            # print(d_k1_hat, d_k1, d_k, v1_hat)

        return [e_u, e_w]
