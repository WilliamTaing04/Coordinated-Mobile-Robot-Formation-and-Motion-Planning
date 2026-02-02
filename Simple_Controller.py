# Simple Controller for jetbot movement
import numpy as np

class control():
    # Initialize the controller.
    def __init__(self, v_max, w_max, deadzone):
        self.v_max = v_max          # max velocity [mm/s]
        self.w_max = w_max          # max angvel [rad/s]
        self.deadzone = deadzone    # deadzone [mm]
        self.motor_max = 0.2       # left motor max speed [0,1]
        self.motor_min = 0.01        # left motor min speed [0,1]
    
    # ORIENTATION X forward, Y left, THETA CCW
    def controller(self, pose, goal):
        # Compute dx, dy, dw
        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        dtheta  = goal[2] - pose[2]

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
            etheta = np.atan2(np.sin(pose[2]), np.cos(pose[2])) # Wrap to -pi,pi
            
            # Gains
            kx = 0.3
            ky = 0.001
            ktheta = 0.001

            # Controller
            v = kx * ex
            w = ky * ey + ktheta * etheta

            # # Only allow for forward movement
            # v = max(0.0, v)

            print(60*"**")
            print(f"pose: {pose}")
            print(f"goal: {goal}")
            print(f"ex:{ex}")
            print(f"ey:{ey}")
            print(f"etheta:{etheta}")
            print(f"v:{v}")
            print(f"w:{w}")

            return v, w
    
    def motor_controller(self, v, w):
        # Normalize v and w(clamped)
        v_n = max(-1, min(v/self.v_max, 1))
        w = max(-self.w_max, min(w, self.w_max))
        w_n = max(-1, min(w/self.w_max, 1))
        
        # Compute left and right motor speeds
        left = v_n - w_n
        right = v_n + w_n

        # Scale motor speeds
        m = max(abs(left), abs(right))
        if m > 1:
            left /= m
            right /= m

        # Max motor speed scale
        left  *= self.motor_max
        right *= self.motor_max

        # Deadbands for motors
        if abs(left) < self.motor_min:
            left = 0.0
        if abs(right) < self.motor_min:
            right = 0.0

        print(f"v_n:{v_n}")
        print(f"w_n:{w_n}")
        print(f"left:{left}")
        print(f"right:{right}")

        return left, right
        
        
        

