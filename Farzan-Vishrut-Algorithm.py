import numpy as np
from math import *
import time as time
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
        _estimator_gains=[-15, -50, -5], #gd, gv, p (NOW POSITIVE)
        _agent_safety_gains=[0.3, 1.4, 1.4], #ds, Eu, Ew
        _extra_parameters=[1,0.1,0.1] #T, dx_star, dy_star
    ):
        self.observed = np.zeros((2,4))
        self.estimated_state = np.zeros((2, 4)) #store dx, vx, dy, vy hat for each agent (both X and Y edges)
        #self.estimated_state = 0.1*np.ones((2,4))
        self.control = np.zeros(2)
        self.id = _id
        self.n_agents = _n_agents
        self.estimator_gains = _estimator_gains
        self.agent_safety_gains = _agent_safety_gains
        self.extra_parameters = _extra_parameters





    def u_w_calculation(self, estimates, observation):
        gd, gv, p = self.estimator_gains
        ds, Eu, Ew = self.agent_safety_gains
        T, dx_star, dy_star = self.extra_parameters

        dx = observation[1,0] # Y edge
        dy = observation[1,1] # Y edge
        v = observation[0,2] # X edge TODO v is the follower velocity, so is the same for both edges?

        v1y_hat = estimates[1,3] # Y edge
        v1x_hat = estimates[0,1] # X edge

        yc = (abs(ds)/ds)*(-gd*(dy_star-ds))-Ew
        xc = -gd*(dx_star-ds) - Eu
        h1 = observation[0,0] - ds - T * v
        alpha = -gd*h1
        k = 1/T

        w = (v1y_hat - gd*(dy-ds)/dx) - (abs(ds)*(Ew+yc))/(ds*dx) #Y edge traits
        print("v1x_hat:",v1x_hat, "v:", v, "alpha:", alpha)
        u = k*(v1x_hat - Eu - xc - v + observation[0,1] * w + alpha) #X edge traits

        control = [u, w]
        return control



    # -------------------------------------------------------------------------------------------
    # INPUT:
    # observation = [dx, dy, v, theta,] of leader to follower
    # estimates = [dx_hat, v1x_hat, dy_hat, v1y_hat] all of which are fed back
    # control = [u, w]
    #
    # OUTPUT:
    # estimates_dot = [dx_hat_dot, v1x_hat_dot, dy_hat_dot, v1y_hat_dot]
    # -------------------------------------------------------------------------------------------
    def estimator_dynamics(self, control, estimates, observation):
        gd, gv, p = self.estimator_gains

        w = control[1]

        dx = observation[0]
        dy = observation[1]
        v = observation[2]

        dx_hat = estimates[0]
        v1x_hat = estimates[1]
        dy_hat = estimates[2]
        v1y_hat = estimates[3]

        state_dot = np.zeros(4)

        dx_del = dx - dx_hat
        dy_del = dy - dy_hat

        state_dot[0] = v1x_hat - v + dy * w + gd * dx_del
        state_dot[1] = gv * dx_del + v1y_hat * w + p * w * dy_del

        state_dot[2] = v1y_hat - dx * w + gd * dy_del
        state_dot[3] = gv * dy_del - v1y_hat * w - p * w * dx_del

        return state_dot

    def system_dynamics(self, estimates, control):
            dynamics = np.zeros((2,4))
            obs = self.observed
            for idx in [0,1]:
                dynamics[idx, :] = self.estimator_dynamics(control, estimates[idx, :], obs[idx, :])
                return dynamics


    '''
    def RK4_step(self, h=0.01)

    *Computes one time step of estimated steps.*
        Input: self.initial_state initial set of estimated states of predecessor for both the X+ and Y edge
        Output: self.estimated_state is updated to the new set of estimated states of predecessor for both the X+ and Y edge
        
    Simplified RK4 step model. Runs with a given timestep (0.01s in this case). The control inputs and self-state 
    (x,y,v,theta) of follower are treated as constant. In the simulation implementation, the state_dynamics were used to
    update state with each step. Take "step" to mean operation between k(n+1) and k(n+1). Future iterations could either
        1) Implement the same state_dynamics to get mathematically extrapolated observed states for use in both the estimator
        dynamics and potentially control_input calculations if those are added too.
        2) We can actually measure the observed states. If the RK4 time step is 10ms, we would need to take measurements at 
        0ms, 5ms, and 10ms for the same calculations listed in (1).
    Now for the control_inputs, if we want to not take them as constant in our RK4 steps, we could
        1) Simply calculate a new control_input with updated state and estimates 
        2) Measure the actual linear acceleration and angular velocity of the robot and use these in the intermediate steps.
        At some point we will need to show the measurements are equivalent to the calculations, and we hope that they will be.
    We may be able to ignore the dynamics of the control and observed state within one time step for either of the following
    reasons:
        1) May be desirable to RK4 for only the variables of interest to be integrated (estimated_states)
        2) The change in the control and observed states may be negligible within one time step.
    '''
    def RK4_step(self, h=0.01):
        # We don't need to update here but I kinda like it
        self.update_self_state()

        initial_state = np.copy(self.estimated_state)

        # Grabs Un
        control_input = self.u_w_calculation(self.estimated_state, self.observed)

        # We don't need to write here but I kinda like it here
        self.write_control(control_input)

        # K1 dependent on Kn and Un
        k1 = self.system_dynamics(initial_state, control_input)

        # Simple updated Xn = S2 for K2 calculation
        s2 = initial_state + k1 * (h/2)

        # K2 dependent on Xn + K1(h/2) and Un + h/2? Not sure about the step in the input
        k2 = self.system_dynamics(s2, control_input)

        s3 = initial_state + k2 * (h/2)

        k3 = self.system_dynamics(s3, control_input)

        s4 = initial_state + k3 * h

        k4 = self.system_dynamics(s4, control_input)

        next_state = initial_state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.estimated_state = np.copy(next_state)

    def write_control(self, control_input):
        print(control_input[0], control_input[1])
        return 0

    '''
    Proposed alternative function to self_dynamics
    '''
    def update_self_state(self):
        # Get observed values from X+ edge and Y edge
        d_X, v_X, theta_X = [0.5, 0, np.pi/8]
        d_Y, v_Y, theta_Y = [0.5, 0, np.pi/8]
        self.observed = np.array([[d_X * cos(theta_X), d_X * sin(theta_X), v_X, theta_X],
                                [d_Y * cos(theta_Y), d_Y * sin(theta_Y), v_Y, theta_Y]])

def run_exp():
    agent = Agent()
    timeStamp = time.perf_counter()
    #Run 1000 steps
    for x in range(500):
            while((time.perf_counter() - timeStamp) < 0.01):
                pass
            agent.RK4_step()
            timeStamp = time.perf_counter()

if __name__ == "__main__":
     run_exp()

