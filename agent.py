import numpy as np
from math import *

WHEEL_SPEED_RPM = 200
WHEEL_RADIUS = 0.05
BOT_RADIUS = 0.08
LINEAR_SPEED = WHEEL_SPEED_RPM * 2 * pi * WHEEL_RADIUS / 60
ANGULAR_SPEED = LINEAR_SPEED / BOT_RADIUS
DR = 0.1

xlim = 2.0
ylim = 2.0
v_max = LINEAR_SPEED  # 0.5
e_max = LINEAR_SPEED  # 0.5
u_max = 0.5
w_max = ANGULAR_SPEED  # 1
t_head = 1.0
color_list = ["r", "g", "b", "k", "c", "p"]


def alpha(h):
    return 10 * h

# We are going to keep this for now because it allows the rk4 iteration to have
# An extrapolated estimate for the self state during iterative control input calculations
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

    theta = observation[1] + observation[2] - state[3] # Relative + absolute - absolute?
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
            state, # state[0] is x, state[1] is y, state[2] is self v, state[3] is alp which is absolute heading angle
            _id,
            X_id,
            Y_id,
            _cluster_size,
            _estimator_gains, #gd, gv, p
            controller,
    ):
        global xlim, ylim, v_max, e_max, u_max, w_max
        if state is None:
            print("Error")
        self.id = _id
        self.X_id = X_id
        self.Y_id = Y_id
        self.n_agents = _cluster_size
        self.estimator_gains = _estimator_gains

        self.cluster_state = np.zeros((_cluster_size, 4))
        self.cluster_state[_id, :] = np.array(state)

        self.controls = [0, 0] #u, w
        self.observations = np.zeros((_cluster_size, 3)) # d, relative angle, global angle theta
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
        self.initialized = False

    # Observations is x y v theta for self
    # Observations is d, phi (relative), theta (absolute) for others
    def init_estimates(self):
        if(self.initialized == False):
            #self.update_edges(observations) TODO: RUN THIS IN JETBOT CONTROL BEFORE INIT ESTIMATES
            for idx in range(self.n_agents):
                if idx == self.id:
                    continue
                self.cluster_state[idx, 0] = self.observations[idx, 0] * cos(
                    self.observations[idx, 1] # Relative x coordinate from self to agent idx d cos(phi)
                )
                self.cluster_state[idx, 2] = self.observations[idx, 0] * sin(
                    self.observations[idx, 1] #Relative y coordinate from self to agent idx d sin(phi)
                )

            self.agent_metadata[0] = self.cluster_state
            self.initialized == True


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

    def RK4_step(self, h=0.0333333):
        global v_max

        initial_state = np.copy(self.cluster_state)

        # Each rk4 iteration, initial_state[self.id, :] must be initialized to x,y,v,theta

        h2 = h / 2
        self.agent_metadata[0] = initial_state
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            # These safety bounds used to be plotted, but what do we do with them now?
            initial_state[self.id, :],
            self.agent_metadata,
        )
        k1 = self.system_dynamics(initial_state, control_input)

        s2 = initial_state + k1 * (h2)
        self.agent_metadata[0] = s2
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            s2[self.id, :],
            self.agent_metadata,
        )
        k2 = self.system_dynamics(s2, control_input)

        s3 = initial_state + k2 * (h2)
        self.agent_metadata[0] = s3
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            s3[self.id, :],
            self.agent_metadata,
        )
        k3 = self.system_dynamics(s3, control_input)

        s4 = initial_state + k3 * h
        self.agent_metadata[0] = s4
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            s4[self.id, :],
            self.agent_metadata,
        )
        k4 = self.system_dynamics(s4, control_input)

        next_state = initial_state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.cluster_state = np.copy(next_state)

        # Clamp velocity
        if self.cluster_state[self.id, 2] < 0:
            self.cluster_state[self.id, 2] = 0
        elif self.cluster_state[self.id, 2] > v_max:
            self.cluster_state[self.id, 2] = v_max

        #Finally, update self.controls internally in this call
        self.agent_metadata[0] = next_state
        control_input, safeh_1, safeh_2 = self.controller.get_control(
            next_state[self.id, :],
            self.agent_metadata,
        )
    def get_controls(self): #We want this to be previous RK4 result with freshly updated observed values
        print(self.controller.controls)
        return self.controller.controls

    def update_edges(self, X_upd, Y_upd):
        self.observations[self.X_id, 0] = X_upd[0]  # Distance to x edge agent
        self.observations[self.X_id, 1] = X_upd[2]  # Relative angle to x edge agent
        #self.observations[X_id, 2] = theta  # self heading angle
        self.observations[self.Y_id, 0] = Y_upd[0]  # Distance to y edge agent
        self.observations[self.Y_id, 1] = Y_upd[2]  # Relative angle to y edge agent
        #self.observations[Y_id, 2] = theta  # self heading angle
        self.agent_metadata[1] = self.observations[:, 0:2].copy()  # Store distance and relative angle