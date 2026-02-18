import numpy as np
from math import *

'''
Question: For U and W controllers in the paper. Do the variables in each refer to 
X+ and Y edge predecessor's respectively?

Test scenario for one leader one follower:
Note 1: The leader will be remote-controlled live or will follow a PID-controlled
trajectory
Note 2: The follower will have its X+ edge and Y edge both set to the leader

Step 1: Initialize Agents 0, and 1. We only need one "group frame" which will be the
follower which sees the leader as "estimate" and itself as "self"

Step 2: loop:
    Step a: Take measurements of self [x, y, v, theta]
        Question: Previous implementation did RK4 iterations on state_dynamics, can we avoid this?
    Step b: Iterate RK4 on estimates [dxhat, v1xhat, dyhat, v1yhat] of leader
    Step c: Calculate control inputs [u, w] and write these to the follower
        Question: How do the U and W controllers differ? When do CBF's get enforced?
'''

class Agent:
    def __init__(
        self,
        _id=0,
        _n_agents = 1,
        _estimator_gains=[15, 50, 5], #gd, gv, p (NOW POSITIVE)
        _agent_safety_gains=[0.3, 1.4, 1.4], #ds, Eu, Ew
        _extra_parameters=[1,0,0] #T, xc, yc
    ):
        self.observed = np.zeros((2,4))
        self.estimated_state = np.zeros((2, 4)) #store dx, vx, dy, vy hat for each agent (both X and Y edges)
        self.control = np.zeros(2)
        self.id = _id
        self.n_agents = _n_agents
        self.estimator_gains = _estimator_gains
        self.agent_safety_gains = _agent_safety_gains
        self.extra_parameters = _extra_parameters

def run_exp():
    agent = Agent()
    state_dot = np.zeros((2, 4))

    agent.observed = None  # get from camera [dx, dy, theta, v] for both X and Y edges

    state_dot[0, :] = agent.estimator_dynamics(agent.control, agent.estimated_state[0, :], agent.observed[0, :])
    state_dot[1, :] = agent.estimator_dynamics(agent.control, agent.estimated_state[1, :], agent.observed[1, :])

    agent.estimated_state = None  # RK4 calculation takes in state_dot

    control = agent.u_w_calculation(agent.estimated_state, agent.observed)

def u_w_calculation(self, estimates, observation):
    gd, gv, p = self.estimator_gains
    ds, Eu, Ew = self.agent_safety_gains
    T, dx_star, dy_star = self.extra_parameters

    dx = observation[1,0] # Y edge
    dy = observation[1,1] # Y edge
    v = observation[0,3] # X edge

    v1y_hat = estimates[1,3] # Y edge
    v1x_hat = estimates[0,1] # X edge

    yc = (abs(ds)/ds)(-gd(dy_star-ds))-Ew
    xc = -gd(dx_star-ds) - Eu
    h1 = dx - ds - T * v
    alpha = -gd*h1
    k = 1/T

    w = [v1y_hat - gd(dy-ds)]/dx - [abs(ds)*(Ew+yc)]/(ds*dx) #Y edge traits
    u = k(v1x_hat - Eu - xc - v + dy * w + alpha * h1) #X edge traits

    control = [u, w]
    return control

'''
Proposed alternative function to self_dynamics
'''
def update_self_state(self):
    # Get observed values from X+ edge and Y edge
    d_X, v_X, theta_X = None
    d_Y, v_Y, theta_Y = None
    self.observedX = np.array([d_X * cos(theta_X), d_X * sin(theta_X), v_X, theta_X])
    self.observedY = np.array([d_Y * cos(theta_X), d_Y * sin(theta_X), v_X, theta_X])
'''
# -------------------------------------------------------------------------------------------
# INPUT:
# observation = [dx, dy, theta, v] of leader to follower
# estimates = [dx_hat, v1x_hat, dy_hat, v1y_hat] all of which are fed back
# control = [u, w]
#
# OUTPUT:
# estimates_dot = [dx_hat_dot, v1x_hat_dot, dy_hat_dot, v1y_hat_dot]
# -------------------------------------------------------------------------------------------
'''
def estimator_dynamics(self, control, estimates, observation):
    gains = self._estimator_gains
    gd = gains[0]
    gv = gains[1]
    p = gains[2]

    w = control[1]

    dx = observation[0]
    dy = observation[1]
    v = observation[3]

    dx_hat = estimates[0]
    v1x_hat = estimates[1]
    dy_hat = estimates[2]
    v1y_hat = estimates[3]

    state_dot = np.zeros((1, 4))

    dx_del = dx - dx_hat
    dy_del = dy - dy_hat

    state_dot[0] = v1x_hat - v + dy * w + gd * dx_del
    state_dot[1] = gv * dx_del + v1y_hat * w + p * w * dy_del

    state_dot[2] = v1y_hat - dx * w + gd * dy_del
    state_dot[3] = gv * dy_del - v1y_hat * w - p * w * dx_del

    return state_dot

def RK4_step(self, h=0.01):
    '''
    Note h2 from prior code has been updated to h/2 for readability
    '''
    #STATE1:
    # Grab initial state vector dx v1x dy v1y
    initial_state = np.copy(self.estimated_state)

    # Grabs initial control input vector
    control_input, safeh_1, safeh_2 = self.controller.get_control(
        initial_state[self.id, :],
        self.agent_metadata,
    )

    # Calculate K1 (dependent on Kn and Un) by solving 4D estimator using the initial_state and control_input
        # K1 = f(tn, xn) = f(xn, Un), since Un carries time dependence
        # xdot = f(x,u) = the 4D estimator equation
    k1 = self.system_dynamics(initial_state, control_input)

    #STATE 2:
    # Update the current state vector with state 2 = initial_state + k1 * (h/2)
        # Cluster state is updated to state 2 because state 2 is the state estimate if you follow
        # the k1 slope for half a time step. This improved state estimate is used to calculate
        # the control inputs which evolve with time the same way the state vector does
    s2 = initial_state + k1 * (h2)

    # Update control input vector to Un + h/2
    control_input, safeh_1, safeh_2 = self.controller.get_control(
        s2[self.id, :],
        self.agent_metadata,
    )
    # Solve K2 
        # Dependent on Xn + K1(h/2) and Un + h/2? Not sure about the step in the input
    
    #STATE 3:
    # Update current state vector to state3 = initial_state + k2 * (h/2)
    
    # Update control input vector to Un + h/2. The time parameter is the same as state 2, but since this input 
    # vector is also dependent on the new state 2, it will be different.     TODO: is control input dependent on state?

    # Solve K3
        # Dependent on Xn + K2(h/2) and Un + h/2
        # f(t+h/2, Xn + K2(h/2))
    
    #STATE 4:
    # Update current state vector to state 4 = initial_state + k3 * h

    # Update control input vector to Un + h 
    
    # Solve K4

    # Finally, update global state to the RK4 result which is:
    # next_state = initial_state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    
    # This may only be important for a simulation with artificial time,
    # in an actual implementation I believe we would read the current time
    # and update it here. Alternatively, since our system is not explicitly
    # time dependent, we may be able to exclude this completely
    self.t_cur += h

    self.init_localizer()
