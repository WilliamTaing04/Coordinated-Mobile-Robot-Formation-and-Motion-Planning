import numpy as np
from math import *
'''
State Variables:
d - Euclidean distance between follower and its predecessor
theta - angle between line-of-sight from follower to predecessor and follower's heading
v1 - predecessor linear velocity
v - follower linear velocity
psi - angle between predecessor's heading and follower's heading

Observable states for the followers are d, theta, v

Control Inputs:
u1 - predecessor linear acceleration
u - follower linear acceleration
w1 - predecessor angular velocity 
w - follower angular velocity

Follower control objective: compute u and w to track the desired inter-agent distance

gd, gv, and p are positive constant gains.
p = gd/3, gv = -2(gd^2) / 9, r = -2p

'''

# Assumed available parameters from AprilTag local measurements: #
#relativeAngle = None 
#distance = None

# Assumed available parameters from overhead or local IMU / Encoder
#u = None # Follower linear acceleration
#w = None # Follower angular velocity

'''
Eq(34) u calculation:
u = k(v1xHat - Eu - xc - v + dy * w + alpha(h1))
eq(27) dictates the safety function along the X+ edge
k = 1/T
    - T defined in h1
v1xHat
    -estimate of v1x, which is (measured?)
Eu
    -v1xhat - Eu = v1x
    -i.e. Eu > 0 is the upper bound of the error
xc
    - non-negative tuning parameter
v = follower linear velocity (measured?)
dy
    - known distance
w = follower angular velocity (measured?)
alpha(h1) = -gd(h1)
h1 = dx - ds - T * v
    - dx is known 
    - ds > 0 is the desired safe distance
    - follower linear velocity (measured?)
    -  T > 0 is the time headway, buffer for reaction time




Eq(45) w calculation:
w = [v1yHat - gd(dy-ds)]/dx - [abs(ds)*(Eω+yc)]/(ds*dx)

xHat = [dxHat, v1xHat, dyHat, v1yHat]^T

v1yHat = dxHat/t 
    - differentiate dxHat for velocity

gd(dy-ds)
    - gd is known
    - dy is known
    - ds is known based on desired safe distance

dx
    - dx is known

Eω + yc
    - Eω is a specific safety bound that can be calculated by: v1y = v1yHat - Eω
    - yc is a non negative tuning constant defined by: 
        yc = (abs(ds)/ds)(-gd(dyStar-ds))-Eω
        - dyStar = desired lateral position which dy should approach and Eq(49) for constraints
    
'''

# The local X-Axis is in the direction of the follower's heading
# The local Y-Axis is the first 2D orthogonal axis CCW from the X-Axis

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
    state_dot = np.zeros((2,4))
    
    agent.observed = None #get from camera [dx, dy, theta, v] for both X and Y edges

    state_dot[0,:] = agent.estimator_dynamics(agent.control, agent.estimated_state[0,:], agent.observed[0,:])
    state_dot[1,:] = agent.estimator_dynamics(agent.control, agent.estimated_state[1,:], agent.observed[1,:])

    agent.estimated_state = None #RK4 calculation takes in state_dot

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



# -------------------------------------------------------------------------------------------
# INPUT:
# observation = [dx, dy, theta, v] of leader to follower
# estimates = [dx_hat, v1x_hat, dy_hat, v1y_hat] all of which are fed back
# control = [u, w]
#
# OUTPUT:
# estimates_dot = [dx_hat_dot, v1x_hat_dot, dy_hat_dot, v1y_hat_dot]
# -------------------------------------------------------------------------------------------
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




