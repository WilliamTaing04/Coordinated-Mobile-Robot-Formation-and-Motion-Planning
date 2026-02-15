import numpy as np
from math import *

# import matplotlib.pyplot as plt
from controller import RandomController, StateFeedbackController
from planner import CooperativeLocalizer

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

'''
agent_metadata = [cluster_state, observations, n_agents, id] # 

Estimates = [dxhat, v1xhat, dyhat, v1yhat]  # Predecessor values
Observation = [d, , ]                       # Size of cluster
State = [x, y, v, alp]                      # Self state, alp is sometimes theta
Control = [u, w]                            # Self linear acceleration, angular velocity
'''


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
    v = state[2] #self v
    w = control[1] #self w
    gd = gains[0]
    gv = gains[1]
    p = gains[2]

    state_dot = np.zeros((1, 4))

    d = observation[0]
    if d == 0:
        return state_dot

    theta = observation[1] + observation[2] - state[3]
    # print(estimates,d*cos(theta),d*sin(theta))
    dx_del = estimates[0] - d * cos(theta) # Backwards? dxhat - dx
    dy_del = estimates[2] - d * sin(theta) #dyhat - dy

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
            x_init = np.random.random() * 2 * xlim - xlim
            y_init = np.random.random() * 2 * ylim - ylim
            v_init = 0.0  # np.random.random() * v_max
            theta_init = np.random.random() * 3.14 * 2
            state = [x_init, y_init, v_init, theta_init]
        self.id = _id
        self.n_agents = _cluster_size
        self.estimator_gains = _estimator_gains

        self.cluster_state = np.zeros((_cluster_size, 4)) #bad practice, stored in agent_metadata too
        self.state = np.array(state)
        self.cluster_state[_id, :] = np.copy(self.state)
        self.cluster_init = False

        self.t_cur = 0
        self.controls = [0, 0]
        self.t_ns = -2.0

        self.observations = np.zeros((_cluster_size, 3))
        self.experiment_history = []
        self.history_stamp = []
        self.agent_metadata = [
            self.cluster_state, #points to the cluster_state array above, if self.cluster_state
                                #is ever reassigned (it is in RK4!) then this points to the old
                                #outdated one
            self.observations,
            self.n_agents,
            self.id,
        ]
        # self.t_cp = 0
        # self.error_estimates = np.ones((1, _cluster_size)) * e_max
        # self.old_observations_cp = np.zeros((_cluster_size, 1))
        # self.t_cp_thres = 1 / _c_freq
        if controller is None:
            self.controller = RandomController(u_max, w_max, v_max, safety=safety)
        else:
            self.controller = controller

        self.localizer = [CooperativeLocalizer() for _ in range(self.n_agents)]

        self.error_history = []
        self.control_history = []
        self.error_stamps = []

    def get_state(self):
        return self.state

    def get_pos(self):
        return self.state[0:2]

    def get_rotation(self):
        theta = self.state[3]
        R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return R

    def get_position_estimates(self):
        position_estimates = []
        R = self.get_rotation()
        for i in range(self.n_agents):
            if i == self.id:
                position_estimates.append([0, 0])
                continue
            position_estimates.append(
                [self.cluster_state[i, 0], self.cluster_state[i, 2]]
            )
        position_estimates = np.array(position_estimates)

        return np.dot(position_estimates, R.T) + self.state[0:2]

    def get_velocity_orientation_estimates(self):
        velocity_estimates = []

        for i in range(self.n_agents):
            if i == self.id:
                velocity_estimates.append(
                    [self.state[2], cos(self.state[3]), sin(self.state[3])]
                )
                continue
            v1 = sqrt(self.cluster_state[i, 3] ** 2 + self.cluster_state[i, 1] ** 2)
            angle = (
                atan2(self.cluster_state[i, 3], self.cluster_state[i, 1])
                + self.state[3]
            )
            velocity_estimates.append(
                [
                    v1,
                    cos(angle),
                    sin(angle),
                ]
            )

        return np.array(velocity_estimates)

    def get_estimates(self):
        position_estimates = self.get_position_estimates()
        velocity_estimates = self.get_velocity_orientation_estimates()
        return np.hstack((position_estimates, velocity_estimates))

    def get_estimation_error(self, t):
        estimation_error = []
        for idx in range(self.n_agents):
            if idx == self.id:
                estimation_error.append([0, 0])
                continue

            estimation_error.append(
                self.localizer[idx].get_error_bound(t, self.observations[idx, 0:2])
            )
        return estimation_error

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
        self.init_localizer()

    def init_localizer(self):
        t = self.t_cur
        v = self.state[2]
        control = self.controls
        for idx in range(self.n_agents):
            if idx == self.id:
                continue
            self.localizer[idx].set_state(
                t,
                v,
                control,
                self.observations[idx, 0:2].copy(),
                self.cluster_state[idx, :],
            )

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
        self.agent_metadata[1] = self.observations[:,0:2].copy()

    def system_dynamics(self, system_state, controls):
        dynamics = np.zeros(system_state.shape)
        state = np.copy(system_state[self.id, :])
        gains = self.estimator_gains
        obs = self.observations
        for idx in range(self.n_agents):
            # For self, update state?
            if idx == self.id:
                dynamics[idx, :] = state_dynamics(state, controls)
                continue
            # For predecessor, update estimator?
            dynamics[idx, :] = estimator_dynamics(
                state, system_state[idx, :], controls, obs[idx, :], gains
            )
        return dynamics

    def RK4_step(self, h=0.01):
        global v_max

        # Grab initial x vector [x,y,v,theta] for self or
        # [dx, v1x, dy, v1y] for others
        initial_state = np.copy(self.cluster_state)

        h2 = h / 2
        t = self.t_cur

        # This is a symptom of the bad cluster_state structure, this is defined above as initial state
        self.agent_metadata[0] = initial_state

        # Grabs Un
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            t,
            initial_state[self.id, :],
            self.agent_metadata,
        )

        # K1 dependent on Kn and Un
        k1 = self.system_dynamics(initial_state, control_input)

        # Simple updated Xn = S2 for K2 calculation
        s2 = initial_state + k1 * (h2)
        # Cluster state is updated to s2 because s2 is the state estimate if you follow
        # the k1 slope for half a time step. This improved state estimate is used to calculate
        # the control inputs which evolve with time the same way the state vector does
        self.agent_metadata[0] = s2
        # Grabs Un + h/2
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            t + h2,
            s2[self.id, :],
            self.agent_metadata,
        )

        # K2 dependent on Xn + K1(h/2) and Un + h/2? Not sure about the step in the input
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

        #since we load the cluster_state in the next line (and the initial state in the
        # RK4 function is grabbed from this spot) this is where the feedback occurs
        next_state = initial_state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.cluster_state = np.copy(next_state)

        #next_state is storing x,y,v,theta for self agent?
        if self.cluster_state[self.id, 2] < 0:
            self.cluster_state[self.id, 2] = 0
        elif self.cluster_state[self.id, 2] > v_max:
            self.cluster_state[self.id, 2] = v_max

        self.state = np.copy(self.cluster_state[self.id, :])

        # This may only be important for a simulation with artificial time,
        #in an actual implementation I believe we would read the current time
        #and update it here. Alternatively, since our system is not explicitly
        #time dependent, we may be able to exclude this completely
        self.t_cur += h

        self.init_localizer()



