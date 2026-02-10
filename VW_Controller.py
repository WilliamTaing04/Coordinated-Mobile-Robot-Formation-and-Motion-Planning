# VW Controller for jetbot movement
import numpy as np
import time

class PID:
    def __init__(self, kp, ki, kd):
        # Gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Errors
        self.integral = 0.0
        self.error_prev = None
    
    def calculate_PID(self, error, dt):
        # Check for bad dt
        if dt <= 0:
            self.integral = 0.0
            self.error_prev = None
            return 0.0
        
        # Accumulate error
        self.integral += error * dt

        # Calculate derivative
        if self.error_prev is not None:
            derivative = (error - self.error_prev) / dt
        else:
            derivative = 0.0
        self.error_prev = error


        # Compute PID output
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)

        return output
        

class control():
    # Initialize the controller.
    def __init__(self, v_max, w_max, freq, pidv, pidw):
        self.v_max = v_max              # max velocity [mm/s]
        self.w_max = w_max              # max angvel [rad/s]
        self.motor_max = 0.14           # left motor max speed [0,1]
        self.motor_min = 0.05           # left motor min speed [0,1]
        self.last_time = time.time()    # save time for dt [s]
        self.dt_max = 1.0 / freq        # max dt [s]
        self.pidv = pidv                # PID for lin vel [mm/s]
        self.pidw = pidw                # PID for ang vel [rad/s]
        self.wheel_rad = 33            # wheel radius [mm]
        self.wheel_len = 120            # length between wheels [mm]

    def controller_vw(self, measured, goal):
        # Calculate dt and update last_time
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # For bad dt
        if dt <= 0.0 or dt >= 3*self.dt_max:
            self.pidv.integral = 0.0
            self.pidv.error_prev = None
            self.pidw.integral = 0.0
            self.pidw.error_prev = None
            return 0.0, 0.0

        # Slice velocities
        v_m , w_m = measured
        v_g , w_g = goal

        # Calculate errors
        e_v = v_g - v_m
        e_w = w_g - w_m

        # PID feedback
        v_fb = self.pidv.calculate_PID(e_v, dt)
        w_fb = self.pidw.calculate_PID(e_w, dt)

        # Combine PID
        v_cmd = v_g + v_fb
        w_cmd = w_g + w_fb

        # Saturation
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))

        return v_cmd, w_cmd

    
    def motor_controller(self, v_cmd, w_cmd):
        wheel_len = self.wheel_len

        # Wheel linear speeds
        v_r = v_cmd + w_cmd * wheel_len / 2
        v_l = v_cmd - w_cmd * wheel_len / 2

        # normalize to motor command
        r_n = (v_r / self.v_max) * self.motor_max
        l_n = (v_l / self.v_max) * self.motor_max

        # Preserve curvature
        m = max(abs(r_n), abs(l_n), 1e-9)
        if m > self.motor_max:
            scale = self.motor_max / m
            r_n *= scale
            l_n *= scale

        # Apply abs max clamp
        r_n = max(-self.motor_max, min(self.motor_max, r_n))
        l_n = max(-self.motor_max, min(self.motor_max, l_n))

        # Deadbands for motors
        if abs(l_n) < self.motor_min:
            l_n = 0.0
        if abs(r_n) < self.motor_min:
            r_n = 0.0
        
        return l_n, r_n
    
    def wrap_to_pi(a):
        """Wrap angle to [-pi, pi]."""
        return (a + np.pi) % (2*np.pi) - np.pi

    def clamp(x, lo, hi):
        return max(lo, min(hi, x))
