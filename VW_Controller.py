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
        self.i_max = 5000


    
    def calculate_PID(self, error, dt):
        # Check for bad dt
        if dt <= 0:
            self.integral = 0.0
            self.error_prev = None
            return 0.0
        
        # Accumulate error
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.i_max, self.i_max)


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
        self.motor_max = 0.7           # left motor max speed [0,1]
        self.motor_min = 0.075           # left motor min speed [0,1]
        self.last_time = time.perf_counter()    # save time for dt [s]
        self.dt_max = 1.0 / freq        # max dt [s]
        self.pidv = pidv                # PID for lin vel [mm/s]
        self.pidw = pidw                # PID for ang vel [rad/s]
        self.wheel_rad = 33            # wheel radius [mm]
        self.wheel_len = 160            # length between wheels [mm]
        self.last_v_cmd = 0             # previous v_cmd
        self.last_w_cmd = 0             # previous w_cmd
        # Filtering
        self.has_filt = False
        self.v_f = 0.0
        self.w_f = 0.0
        self.alpha = 0.1   # 0.1–0.3 typical
        # Motor Calibration
        self.K_V = 1628.7734269380087
        self.B_V = -107.70507007613396


    def controller_vw(self, measured, goal):
        # Calculate dt and update last_time
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now

        # For bad dt
        if dt <= 0.0 or dt >= 3*self.dt_max:
            self.pidv.integral = 0.0
            self.pidv.error_prev = None
            self.pidw.integral = 0.0
            self.pidw.error_prev = None
            self.has_filt = False
            self.last_time = now
            return self.last_v_cmd, self.last_w_cmd


        # Slice velocities
        v_m , w_m = measured
        v_g , w_g = goal

        # Filter v_m w_m
        if not self.has_filt:
            self.v_f = float(v_m)
            self.w_f = float(w_m)
            self.has_filt = True
        else:
            a = self.alpha
            self.v_f = a*float(v_m) + (1-a)*self.v_f
            self.w_f = a*float(w_m) + (1-a)*self.w_f
        v_m, w_m = self.v_f, self.w_f


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

        self.last_v_cmd = v_cmd
        self.last_w_cmd = w_cmd

        return v_cmd, w_cmd

    
    def motor_controller(self, v_cmd, w_cmd):
        wheel_len = self.wheel_len

        # Wheel linear speeds
        v_r = v_cmd + w_cmd * wheel_len / 2
        v_l = v_cmd - w_cmd * wheel_len / 2

        # Apply calibration
        r_n = (v_r - self.B_V) / self.K_V
        l_n = (v_l - self.B_V) / self.K_V

        # Preserve curvature
        m = max(abs(r_n), abs(l_n), 1e-9)
        if m > self.motor_max:
            scale = self.motor_max / m
            r_n *= scale
            l_n *= scale

        # Apply abs max clamp
        r_n = max(-self.motor_max, min(self.motor_max, r_n))
        l_n = max(-self.motor_max, min(self.motor_max, l_n))

        # Deadband compensation
        if abs(v_cmd) < 5.0 and abs(w_cmd) < 0.05:
            l_n = r_n = 0.0
        else:
            l_n = deadband_comp(l_n, self.motor_min)
            r_n = deadband_comp(r_n, self.motor_min)

        
        return l_n, r_n

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def deadband_comp(u, umin):
    if u == 0.0:
        return 0.0
    s = 1.0 if u > 0 else -1.0
    return s*umin if abs(u) < umin else u
