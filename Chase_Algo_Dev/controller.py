import numpy as np
from math import *

EW = 1.4
EU = 1.4
DR = 0.3
T = 1.0
ALPHA = 5.0
INF = 100

WHEEL_SPEED_RPM = 200
WHEEL_RADIUS = 0.05
BOT_RADIUS = 0.08
LINEAR_SPEED = WHEEL_SPEED_RPM * 2 * pi * WHEEL_RADIUS / 60
ANGULAR_SPEED = LINEAR_SPEED / BOT_RADIUS

V_MAX = LINEAR_SPEED  # 0.5
U_MAX = 0.5
W_MAX = 2.0  # ANGULAR_SPEED  # 1


def safety_candidate_u(state, observation, estimate, err=0):
    # returns bound and a flag(1: upper bound, -1: lower bound, 0: no bound)
    d = observation[0]
    theta = observation[1]
    v = state[2]
    v_1_hat = sqrt(estimate[1] ** 2 + estimate[3] ** 2)
    phi_hat = atan2(estimate[3], estimate[1])
    h = d - DR - T * v * cos(theta)
    LHS_bound = v_1_hat * cos(theta - phi_hat) - err
    LHS_bound -= EU
    LHS_bound -= v
    LHS_bound += ALPHA * h
    LHS_bound /= T
    if cos(theta) == 0:
        return INF, 0
    # print("[U-SAFE] : ", h, LHS_bound, cos(theta))

    return LHS_bound / cos(theta), cos(theta) / abs(cos(theta))


def safety_candidate_w(state, observation, estimate, err=0):
    # returns bound and a flag(1: upper bound, -1: lower bound, 0: no bound)
    d = observation[0]
    theta = observation[1]
    v = state[2]
    v_1_hat = sqrt(estimate[1] ** 2 + estimate[3] ** 2)
    phi_hat = atan2(estimate[3], estimate[1])
    h = 1 - cos(theta)

    LHS_bound = v * sin(theta)
    LHS_bound = LHS_bound - (v_1_hat * sin(theta - phi_hat) - err)
    LHS_bound *= sin(theta)
    LHS_bound -= abs(EW * sin(theta))
    LHS_bound /= d
    LHS_bound += h / T

    # print("[SC-W] : ", h, d, LHS_bound, sin(theta))

    if sin(theta) == 0:
        return LHS_bound, 0

    return LHS_bound / sin(theta), sin(theta) / abs(sin(theta))


def state_dynamics(state, control):
    v = state[2]
    alp = state[3]
    state_dot = np.zeros((1, 4))
    state_dot[0, 0] = v * cos(alp)
    state_dot[0, 1] = v * sin(alp)
    state_dot[0, 2] = control[0]
    state_dot[0, 3] = control[1]
    return state_dot


class Controller:
    def __init__(self, u_max=0.25, w_max=1.0, v_max=0.5, safety=False):
        self.u_max = u_max
        self.w_max = w_max
        self.v_max = v_max
        self.v_min = 0
        self.u_min = -1 * u_max
        self.w_min = -1 * w_max
        self.safety = safety
        bounds = [[self.u_min, self.u_max], [self.w_min, self.w_max]]
        print("[CONTROL] bounds for agent ", id, " : ", bounds)

    def set_safety(self, safety):
        self.safety = safety

    def clip_controls(self, control, state, bounds=None):
        if bounds is not None:
            w_ub = bounds[1][1]
            w_lb = bounds[1][0]
            u_ub = bounds[0][1]
            u_lb = bounds[0][0]
        else:
            w_ub = self.w_max
            w_lb = self.w_min
            u_ub = self.u_max
            u_lb = self.u_min

        clipped_control = [0, 0]
        clipped_control[0] = min(max(u_lb, control[0]), u_ub)
        w_allowed = self.get_velocity_bound(v=state[2])
        w_ub = min(w_ub, w_allowed)
        w_lb = max(w_lb, -1 * w_allowed)
        clipped_control[1] = min(max(w_lb, control[1]), w_ub)
        if clipped_control[1] != control[1]:
            clipped_control[0] = min(clipped_control[0], 0)
        return clipped_control

    def get_velocity_bound(self, v=None, w=None):
        if v is not None:
            v = max(0, min(v, self.v_max))
            return (self.v_max - v) * self.w_max / self.v_max

        if w is not None:
            w = min(abs(w), self.w_max)
            return self.v_max - (self.v_max * w / self.w_max)

        raise ValueError

    def get_safety_controls_bound(
        self, control, state, agent_metadata, estimation_error=None
    ):
        bounds = [[self.u_min, self.u_max], [self.w_min, self.w_max]]
        estimates, observations, n_agents, id = tuple(agent_metadata)
        if estimation_error is None:
            estimation_error = [[0, 0] for _ in range(n_agents)]
        for i in range(n_agents):
            if i == id:
                continue
            e_u = estimation_error[i][0]
            e_w = estimation_error[i][1]
            u_bound, flag_u = safety_candidate_u(
                state, observations[i, :], estimates[i, :], err=e_u
            )
            w_bound, flag_w = safety_candidate_w(
                state, observations[i, :], estimates[i, :], err=e_w
            )
            if flag_u == 1 and u_bound <= bounds[0][1]:
                bounds[0][1] = u_bound
                if control[0] >= u_bound:
                    if flag_w == 1:
                        bounds[1][1] = min(bounds[1][1], w_bound)
                    elif flag_w == -1:
                        bounds[1][0] = max(bounds[1][0], w_bound)

            elif flag_u == -1 and u_bound >= bounds[0][0]:
                bounds[0][0] = u_bound
                if control[0] <= u_bound:
                    if flag_w == 1:
                        bounds[1][1] = min(bounds[1][1], w_bound)
                    elif flag_w == -1:
                        bounds[1][0] = max(bounds[1][0], w_bound)

        if bounds[0][1] < bounds[0][0] or bounds[1][1] < bounds[1][0]:
            bounds[0][1] = -0.25
            bounds[0][0] = -0.25
            # print("[SAFETY] bounds for agent ", id, " : ", bounds)
        return bounds

    def get_control(
        self,
        t=0,
        state=None,
        desired_state=None,
        agent_metadata=None,
        estimation_error=None,
    ):
        controls = self.calc_control(t, state, desired_state)
        bounds = None
        if self.safety:
            if agent_metadata is None:
                raise ValueError
            bounds = self.get_safety_controls_bound(
                controls, state, agent_metadata, estimation_error
            )
        return self.clip_controls(controls, state, bounds)

    def calc_control(self, t=0, state=None, desired_state=None):
        raise NotImplementedError()


class RandomController(Controller):
    def __init__(self, u_max=0.25, w_max=1.0, v_max=0.5, safety=False):
        super().__init__(u_max, w_max, v_max, safety)
        self.controls = [0, 0]
        self.t_ns = 0

    def calc_control(self, t=0, state=None, desired_state=None):
        if t - self.t_ns > 2.0:
            u = np.random.random() * 2 * self.u_max - self.u_max
            w = np.random.random() * 2 * self.w_max - self.w_max
            self.controls = [u, w]
            self.t_ns = t
        return self.controls


class StateFeedbackController(Controller):
    def __init__(self, u_max=0.25, w_max=1.0, v_max=0.5, safety=False):
        super().__init__(u_max, w_max, v_max, safety)
        self.controls = [0, 0]
        self.t_ns = 0
        self.gains = {"kw": 0.6, "kv": 1.0, "kr": 0.6}

    def set_gains(self, kw, kv, kr):
        self.gains["kw"] = abs(kw)
        self.gains["kv"] = abs(kv)
        self.gains["kr"] = abs(kr)

    def calc_control(self, t=0, state=None, desired_state=None):
        e_r, e_v, e_alp = StateFeedbackController.state_feedback_errors(
            state, desired_state
        )

        w = self.gains["kw"] * e_alp
        u = self.gains["kv"] * e_v + self.gains["kr"] * e_r

        self.controls = [u, w]
        self.t_ns = t
        return self.controls

    @staticmethod
    def state_feedback_errors(state, desired_state):
        # print(state, desired_state)
        e_x = desired_state[0] - state[0]
        e_y = desired_state[1] - state[1]
        e_v = desired_state[2] - state[2]
        r = sqrt(e_x**2 + e_y**2)
        if r <= 0.1:
            r = 0

        val = exp(-10 * r)
        theta = atan2(e_y, e_x)
        alp = state[3]
        e_r = r * (cos(theta - alp) ** 2)
        desired_alp = desired_state[3] * val + theta * (1 - val)
        e_alp = desired_alp - alp
        while e_alp > pi:
            e_alp -= 2 * pi
        while e_alp < -1 * pi:
            e_alp += 2 * pi

        # print(e_r, e_v, e_alp)
        return e_r, e_v, e_alp


class SimpleController:
    def get_control(self, t, state, agent_metadata):
        controls, out1, out2 = self.calc_control(t, state, agent_metadata)
        clipped_controls = self.clip_control(controls)
        return clipped_controls, out1, out2

    def clip_control(self, control):
        clipped_control = [0, 0]
        clipped_control[0] = min(max(-1 * U_MAX, control[0]), U_MAX)
        clipped_control[1] = min(max(-1 * W_MAX, control[1]), W_MAX)
        return clipped_control

    def calc_control(self, t, state, agent_metadata):
        raise NotImplementedError


class SafeFormationController(SimpleController):
    def __init__(self, float_lst):
        x_id = int(float_lst[0])
        y_id = int(float_lst[1])
        ds_x = float_lst[2]
        ds_y = float_lst[3]
        self.dsafe_x = DR
        self.dsafe_y = float_lst[4]
        gd = float_lst[5]
        self.controls = [0, 0]
        self.sgn_s = 1
        if self.dsafe_y != 0:
            self.sgn_s = self.dsafe_y / abs(self.dsafe_y)

        self.xc = max(-1 * gd * (ds_x - self.dsafe_x) - EU, 0)
        self.yc = self.sgn_s * (-1 * gd * (ds_y - self.dsafe_y)) - EW
        self.yc = max(self.yc, 0)
        self.gd = gd
        self.ds_x = ds_x
        self.ds_y = ds_y
        self.x_id = x_id
        self.y_id = y_id

    def calc_control(self, t, state, agent_metadata):
        cluster_state = agent_metadata[0]
        observations = agent_metadata[1]
        v1x_hat = cluster_state[self.x_id, 1]
        v1y_hat = cluster_state[self.y_id, 3]
        v = state[2]
        dx = observations[self.x_id, 0] * cos(observations[self.x_id, 1])
        obs_dy = observations[self.y_id, 0]
        if obs_dy < self.ds_y:
            obs_dy = self.ds_y
        dy = obs_dy * sin(observations[self.y_id, 1])

        h2 = (dy - self.dsafe_y) * self.sgn_s

        w = (v1y_hat - self.gd * (dy - self.dsafe_y) - self.sgn_s * (self.yc + EW)) / dx

        h1 = dx - self.dsafe_x - T * v
        alpha_h = -1 * self.gd * h1
        u = (v1x_hat - EU - self.xc - v + dy * w + alpha_h) / T

        self.controls = [u, w]
        if h1 < 0 or h2 < 0:
            print("[INFO] CONTROLS:", self.controls)
            print(
                "[ERROR] h1: ",
                h1,
                " h2: ",
                h2,
                " dx: ",
                dx,
                " v: ",
                v,
                " d_safe_x: ",
                self.dsafe_x,
                " dy: ",
                dy,
            )
        return [u, w], dx - T * v, dy * self.sgn_s


class SinosodialController(SimpleController):
    def __init__(self, float_list):
        self.t = 0
        self.f = 0.5 * 2 * pi
        self.amp = float_list[0]

    def calc_control(self, t, state, agent_metadata):
        return [self.amp * sin(t * self.f), 0], 0, 0


class ConstantController(SimpleController):
    def __init__(self, float_list):
        self.controls = float_list

    def calc_control(self, t, state, agent_metadata):
        if t > 2.0:
            self.controls[0] = 0.0
        
        return self.controls, 0, 0


class TrajectoryController(SimpleController):
    def __init__(self, float_lst):
        self.t = 0
        self.t_step = float_lst[0]
        self.x_traj = [1, 1.5, 2, 3, 3.5, 4, 4, 3, 1.5, 0, 0]
        self.y_traj = [0, 0.5, 1, 1, 0.5, 0, -1, -2, -1, 0, 0]
        self.theta_traj = [0, 45, 0, 0, -45, -90, -90, -180, 135, 90, 0]
        self.t_max_idx = len(self.x_traj) - 1
        self.controller = StateFeedbackController()

    def calc_control(self, t, state, agent_metadata):
        self.t = t
        idx = int(t // self.t_step)
        if idx > self.t_max_idx:
            idx = self.t_max_idx
        xd = self.x_traj[idx]
        yd = self.y_traj[idx]
        vd = 0
        td = self.theta_traj[idx] * pi / 180
        desired_state = [xd, yd, vd, td]
        controls = self.controller.calc_control(t, state, desired_state)
        #if abs(controls[1]) > 0.1:
        #    controls[1] = 0.1* controls[1]/abs(controls[1])
        return controls, 0, 0


# class Trash():
##def coop_localize(self):
##    global a_max
##    del_t = self.t_cur - self.t_cp
##    if del_t > self.t_cp_thres:
##        v = self.state[2]
##        theta = self.observations[:,1]
##        del_d = self.observations[:,0] - self.old_observations_cp
##        self.error_estimates = del_d/del_t + v*cos(theta)

# def safety_control(self,state=None):
#    global t_head, d_r, u_max
#    if state is None:
#        state = self.cluster_state
#    v = state[self.id,2]
#    theta = self.observations[:,1]
#    v1_hat = np.sqrt(state[:,1]**2 + state[:,3]**2)
#    phi_hat = np.arctan2(state[:,3],state[:,1])
#    h = self.observations[:,0] - d_r - t_head*v

#    limits = ((v1_hat*np.cos(theta-phi_hat) + self.error_estimates - v*np.cos(theta) + alpha(h))/t_head).T
#    limits[self.id] = u_max
#    return max(min(limits),-1*u_max)
