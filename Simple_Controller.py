# Simple Controller for jetbot movement
import numpy as np

class control():
    # Initialize the controller.
    def __init__(self, v_max, w_max, deadzone):
        self.v_max = v_max          # max velocity [mm/s]
        self.v_min = 0.1          # min velocity [mm/s]
        self.w_max = w_max          # max angvel [rad/s]
        self.w_min = 0.1          # min angvel [rad/s]
        self.deadzone = deadzone    # deadzone [mm]
        self.len = 12               # length between wheels [mm]
        self.rad = 3.3              # radius of wheels [mm]
        self.left_max = 0.2         # motor max speed [0,1]
        self.right_max = 0.2        # motor max speed [0,1]
    

    def controller(self, pose, goal):
        # Compute dx, dy, dw
        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        dtheta = goal[2] - pose[2]

        # Find absolute distance to goal
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # If within deadzone then stop
        if dist < self.deadzone:
            return 0.0, 0.0
        
        # If dist > deadzone
        else:
            # errors
            ex = dx * np.cos(pose[2]) + dy * np.sin(pose[2])
            ey = -dx * np.sin(pose[2]) + dy * np.cos(pose[2])
            etheta = np.atan2(np.sin(dtheta), np.cos(dtheta)) # Wrap to -pi,pi
            
            # Gains
            kx = 0.1
            ky = 0.1
            ktheta = 0.1

            # Controller
            v = kx * ex
            w = ky * ey + ktheta * etheta

            # # Only allow for forward movement
            # v = max(0.0, v)

            return v, w
    
    def motor_controller(self, v, w):
        # Normalize v and w
        v_n = max(-1, min(v/self.v_max, 1))
        w_n = max(-1, min(w/self.w_max, 1))


        # Deadbands for motors
        if abs(v_n) < self.v_min:
            v_n = 0.0
        if abs(w_n) < self.w_min:
            w_n = 0.0
        if v_n == 0.0 and w_n == 0.0:
            return 0.0, 0.0
        
        # Compute left and right motor speeds
        left = v - w
        right = v + w

        # Scale motor speeds
        m = max(abs(left), abs(right))
        if m > 1:
            left /= m
            right /= m

        # Max motor speed scale
        left  *= self.left_max
        right *= self.right_max

        return left, right
        
        
        

