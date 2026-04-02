import numpy as np
from math import *

# import matplotlib.pyplot as plt
from controller import RandomController, StateFeedbackController

WHEEL_SPEED_RPM = 200
WHEEL_RADIUS = 0.05
BOT_RADIUS = 0.08
LINEAR_SPEED = WHEEL_SPEED_RPM * 2 * pi * WHEEL_RADIUS / 60
ANGULAR_SPEED = LINEAR_SPEED / BOT_RADIUS
DR = 0.3

xlim = 2.0
ylim = 2.0
v_max = LINEAR_SPEED  # 0.5
e_max = LINEAR_SPEED  # 0.5
u_max = 0.5
w_max = ANGULAR_SPEED  # 1
t_head = 1.0
d_r = 0.5
color_list = ["r", "g", "b", "k", "c", "p"]


def alpha(h):
    return 10 * h


def state_dynamics(state, control):
    v = state[2]
    alp = state[3]
    state_dot = np.zeros((1, 4))
    state_dot[0, 0] = v * cos(alp)
    state_dot[0, 1] = v * sin(alp)
    state_dot[0, 2] = control[0]
    state_dot[0, 3] = control[1]
    return state_dot


def estimator_dynamics(state, estimates, control, observation, gains):
    v = state[2]
    w = control[1]
    gd = gains[0]
    gv = gains[1]
    p = gains[2]

    state_dot = np.zeros((1, 4))

    d = observation[0]
    if d == 0:
        return state_dot

    theta = observation[1] + observation[2] - state[3]
    # print(estimates,d*cos(theta),d*sin(theta))
    dx_del = estimates[0] - d * cos(theta)
    dy_del = estimates[2] - d * sin(theta)

    state_dot[0, 0] = estimates[1] - v + d * w * sin(theta) + gd * dx_del
    state_dot[0, 1] = gv * dx_del + estimates[3] * w + p * w * dy_del

    state_dot[0, 2] = estimates[3] - d * w * cos(theta) + gd * dy_del
    state_dot[0, 3] = gv * dy_del - estimates[1] * w - p * w * dx_del

    return state_dot


class Agent:
    def __init__(
            self,
            state=None,
            _id=0,
            _cluster_size=1,
            _estimator_gains=[-15, -50, -5],
            _c_freq=5,
            safety=False,
            controller=None,
    ):
        global xlim, ylim, v_max, e_max, u_max, w_max
        if state is None:
            print("Error")
        self.id = _id
        self.n_agents = _cluster_size
        self.estimator_gains = _estimator_gains

        self.cluster_state = np.zeros((_cluster_size, 4))
        self.state = np.array(state)
        self.cluster_state[_id, :] = np.copy(self.state)
        self.cluster_init = False

        self.t_cur = 0
        self.controls = [0, 0]
        self.t_ns = -2.0

        self.observations = np.zeros((_cluster_size, 3))
        self.agent_metadata = [
            self.cluster_state,
            self.observations,
            self.n_agents,
            self.id,
        ]
        if controller is None:
            print("ERROR")
        else:
            self.controller = controller

    def get_state(self):
        return self.state

    def get_pos(self):
        return self.state[0:2]

    def get_rotation(self):
        theta = self.state[3]
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return R

    def init_estimates(self, observations):
        self.set_observations(observations)

        for idx in range(self.n_agents):
            if idx == self.id:
                continue
            self.cluster_state[idx, 0] = self.observations[idx, 0] * cos(
                self.observations[idx, 1]
            )
            self.cluster_state[idx, 2] = self.observations[idx, 0] * sin(
                self.observations[idx, 1]
            )

        self.agent_metadata[0] = self.cluster_state
        self.t_cur = 0

    def set_observations(self, observations):
        theta = self.state[3]
        for _id, pos in enumerate(observations):
            diff = pos - self.state[0:2]
            self.observations[_id, 0] = sqrt(diff[0] ** 2 + diff[1] ** 2)
            self.observations[_id, 1] = atan2(diff[1], diff[0]) - theta
            self.observations[_id, 2] = theta
        if not self.cluster_init:
            self.cluster_init = True
            for _id, pos in enumerate(observations):
                if _id == self.id:
                    continue
                diff = pos - self.state[0:2]
                d = sqrt(diff[0] ** 2 + diff[1] ** 2)
                phi = atan2(diff[1], diff[0]) - theta
                self.cluster_state[_id, 0] = d * cos(phi)
                self.cluster_state[_id, 2] = d * sin(phi)
        self.agent_metadata[1] = self.observations[:, 0:2].copy()

    def system_dynamics(self, system_state, controls):
        dynamics = np.zeros(system_state.shape)
        state = np.copy(system_state[self.id, :])
        gains = self.estimator_gains
        obs = self.observations
        for idx in range(self.n_agents):
            if idx == self.id:
                dynamics[idx, :] = state_dynamics(state, controls)
                continue
            dynamics[idx, :] = estimator_dynamics(
                state, system_state[idx, :], controls, obs[idx, :], gains
            )
        return dynamics

    def RK4_step(self, h=0.01):
        global v_max

        initial_state = np.copy(self.cluster_state)

        h2 = h / 2
        t = self.t_cur
        self.agent_metadata[0] = initial_state
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            # These safety bounds used to be plotted, but what do we do with them now?
            t,
            initial_state[self.id, :],
            self.agent_metadata,
        )
        k1 = self.system_dynamics(initial_state, control_input)

        s2 = initial_state + k1 * (h2)
        self.agent_metadata[0] = s2
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            t + h2,
            s2[self.id, :],
            self.agent_metadata,
        )
        k2 = self.system_dynamics(s2, control_input)

        s3 = initial_state + k2 * (h2)
        self.agent_metadata[0] = s3
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            t + h2,
            s3[self.id, :],
            self.agent_metadata,
        )
        k3 = self.system_dynamics(s3, control_input)

        s4 = initial_state + k3 * h
        self.agent_metadata[0] = s4
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            t + h,
            s4[self.id, :],
            self.agent_metadata,
        )
        k4 = self.system_dynamics(s4, control_input)

        next_state = initial_state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # print('initial_state',initial_state)
        # print('next_state',next_state)

        self.cluster_state = np.copy(next_state)

        if self.cluster_state[self.id, 2] < 0:
            self.cluster_state[self.id, 2] = 0
        elif self.cluster_state[self.id, 2] > v_max:
            self.cluster_state[self.id, 2] = v_max

        self.state = np.copy(self.cluster_state[self.id, :])
        self.t_cur += h

