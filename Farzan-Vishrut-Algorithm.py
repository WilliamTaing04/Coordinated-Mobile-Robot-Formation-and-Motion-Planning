import numpy as np
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
relativeAngle = None 
distance = None

# Assumed available parameters from overhead or local IMU / Encoder
u = None # Follower linear acceleration
w = None # Follower angular velocity

'''
Eq(34) u calculation:
u = k(v1xHat - Eu - xc - v + dy * w + alpha * h1)
eq(27) dictates the safety function along the X+ edge
k = 1/T
    - T defined in h1
v1xHat 
Eu
xc
    - non-negative tuning parameter
v
dy
    - known distance
w
alpha
h1 = dx - ds - T * v
    - dx is known 
    - ds > 0 is the desired safe distance
    - v is observable and known
    -  T > 0 is the time headway, buffer for reaction time




Eq(45) w calculation:
w = [v1yHat - gd(dy-ds)]/dx - [abs(ds)*(Ew+yc)]/(ds*dx)

xHat = [dxHat, v1xHat, dyHat, v1yHat]^T

v1yHat = dxHat/t 
    - differentiate dxHat for velocity

gd(dy-ds)
    - gd is known
    - dy is known
    - ds is known based on desired safe distance

dx
    - dx is known

Ew + yc
    - Ew is a specific safety bound that can be calculated by: v1y = v1yHat - Ew
    - yc is a non negative tuning constant defined by: 
        yc = (abs(ds)/ds)(-gd(dyStar-ds))-Ew
        - dyStar = desired lateral position which dy should approach and Eq(49) for constraints
    
'''

# The local X-Axis is in the direction of the follower's heading
# The local Y-Axis is the first 2D orthogonal axis CCW from the X-Axis
leaderX = None
followerX = None
leaderY = None
followerY = None


dx = leaderX - followerX #verify for direction of camera x and y
dy = leaderY - followerY
theta = np.atan2(dy,dx) # Angle from follower to leader

w = [v1yHat - gd(dy-ds)]/dx - [abs(ds)*(Ew+yc)]/(ds*dx)




robotOffset = None





