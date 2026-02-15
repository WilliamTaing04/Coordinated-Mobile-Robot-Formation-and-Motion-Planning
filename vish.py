'''
Test scenario for one leader one follower:
Note 1: The leader will be remote-controlled live or will follow a PID-controlled
trajectory
Note 2: The follower will have its X+ edge and Y edge both set to the leader

Step 1: Initialize Agents 0, and 1. We only need one "group frame" which will be the
follower which sees the leader as "estimate" and itself as "self"

Step 2: loop:
    Step a: Take measurements of self [x, y, v, theta]
    Step b: Iterate RK4 on estimates [dxhat, v1xhat, dyhat, v1yhat] of follower
    Step c: Calculate control inputs [u, w] and write these to the follower
    Step d:
'''


class Agent:
    def __init__(
        self,
        _id=0,
        _n_agents = 1,
        _estimator_gains=[-15, -50, -5] #gd, gv, p
    ):
        self.self_state = np.zeros(4)
        self.estimated_state = np.zeros((_n_agents, 4)) #store dx, dvx, dy, dvy for each agent
        self.id = _id
        self.n_agents = _n_agents
        self.estimator_gains = _estimator_gains

def self_dynamics(state, control): # Why does self need to be a dynamic state? Can't we measure x, y, v, theta directly?
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

def RK4_step(self, h=0.01):
    '''
    Note h2 from prior code has been updated to h/2 for readability
    '''

    '''
    STATE1:
    # Grab initial state vector dx v1x dy v1y

    # Grabs initial control input vector

    # Calculate K1 (dependent on Kn and Un) by solving 4D estimator using the initial_state and control_input
        # K1 = f(tn, xn) = f(xn, Un), since Un carries time dependence
        # xdot = f(x,u) = the 4D estimator equation

    STATE 2:
    # Update the current state vector with state 2 = initial_state + k1 * (h/2)
        # Cluster state is updated to state 2 because state 2 is the state estimate if you follow
        # the k1 slope for half a time step. This improved state estimate is used to calculate
        # the control inputs which evolve with time the same way the state vector does
    
    # Update control input vector to Un + h/2

    # Solve K2 
        # Dependent on Xn + K1(h/2) and Un + h/2? Not sure about the step in the input
    
    STATE 3:
    # Update current state vector to state3 = initial_state + k2 * (h/2)
    
    # Update control input vector to Un + h/2. The time parameter is the same as state 2, but since this input 
    # vector is also dependent on the new state 2, it will be different.     TODO: is control input dependent on state?

    # Solve K3
        # Dependent on Xn + K2(h/2) and Un + h/2
    
    STATE 4:
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
    '''
