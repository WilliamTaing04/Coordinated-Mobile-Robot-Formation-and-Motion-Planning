import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags

class Jetbot():
    def __init__(self, id, role=0, tau_pose=0.2, tau_vel=0.0):
        self.id = id
        self.role = role                # 0-follower 1-leader
        self.visible = 0                # 0-not seen 1-seen
        self.time_meas = None           # time of measurement
        self.prev_time_meas = None      # time of previous measurement
        self.pose = None                # [x, y, theta]
        self.prev_pose = None           # [x, y, theta]
        self.pose_f = None              # [x, y, theta] (filtered)
        self.prev_pose_f = None         # [x, y, theta] (filtered)
        self._yaw_unwrapped = None      # [rad] yaw/theta (unwrapped)
        self._prev_yaw_raw = None       # [rad] yaw/theta (wrapped)
        self.lin_vel = None             # mm/s
        self.prev_lin_vel = 0          # mm/s
        self.ang_vel = None             # rad/s
        self.prev_ang_vel = 0          # rad/s
        self.lin_vel_f = 0.0            # mm/s (filtered)
        self.ang_vel_f = 0.0            # rad/s (filtered)
        self._prev_lin_vel_f = 0.0      # mm/s (filtered)
        self._prev_ang_vel_f = 0.0      # rad/s (filtered)
        self.lin_acc = None             # mm/s^2
        self.prev_lin_acc = 0           # mm/s^2
        self.ang_acc = None             # rad/s^2
        self.prev_ang_acc = 0           # rad/s^2
        self.tau_pose = tau_pose        # sec
        self.tau_vel  = tau_vel         # 0 disables extra vel filtering

    def update_meas(self, pose, time_meas):
        pose = np.asarray(pose, dtype=float).copy()
        time_meas = float(time_meas)
        # Check for first update
        if self.pose is None or self.time_meas is None:
            self.pose = pose
            self.prev_pose = pose.copy()
            self._prev_yaw_raw = pose[2]
            self._yaw_unwrapped = pose[2]
            self.pose_f = pose.copy()
            self.prev_pose_f = pose.copy()
            self.time_meas = time_meas
            self.prev_time_meas = time_meas

            self.lin_vel = 0.0
            self.ang_vel = 0.0
            self.lin_acc = 0.0
            self.ang_acc = 0.0
            return
        
        # Update time measured
        self.prev_time_meas = self.time_meas
        self.time_meas = time_meas
        dt = self.time_meas - self.prev_time_meas

        # Check for dt contraints
        if dt <= 1e-6:
            return
        if dt > 0.2:   # ignore dt > 200ms 
            # Update pose/time
            self.prev_pose = self.pose
            self.pose = pose
            self.prev_pose_f = self.pose_f.copy()
            self.pose_f = pose.copy() 
            self._prev_yaw_raw = pose[2]
            self._yaw_unwrapped = pose[2]
            self.lin_vel = 0.0
            self.ang_vel = 0.0
            self.lin_vel_f = 0.0
            self.ang_vel_f = 0.0
            self.lin_acc = 0.0
            self.ang_acc = 0.0
            return

        # Update pose
        self.prev_pose = self.pose # what is this used for
        self.prev_lin_vel = self.lin_vel
        self.prev_ang_vel = self.ang_vel
        self.prev_pose_f = self.pose_f.copy()

        self.pose = pose 

        # Unwrap yaw
        dyaw = self.wrap_to_pi(self.pose[2] - self._prev_yaw_raw)
        self._yaw_unwrapped = self._yaw_unwrapped + dyaw
        self._prev_yaw_raw = self.pose[2] # what is this for
        
        # LPF pose
        alpha_pose = dt / (self.tau_pose + dt) if self.tau_pose > 0 else 1.0
        

        # filter x,y
        self.pose_f[0] = (1 - alpha_pose) * self.pose_f[0] + alpha_pose * self.pose[0] # X
        self.pose_f[1] = (1 - alpha_pose) * self.pose_f[1] + alpha_pose * self.pose[1] # Y
        # filter unwrapped yaw
        self.pose_f[2] = (1 - alpha_pose) * self.pose_f[2] + alpha_pose * self._yaw_unwrapped # is this correct?

        # Find displacement from last move
        dx = self.pose_f[0] - self.prev_pose_f[0]
        dy = self.pose_f[1] - self.prev_pose_f[1]

        # Calculate velocities using filtered values
        yaw_prev = self.prev_pose_f[2]
        self.lin_vel = (dx * np.cos(yaw_prev) + dy * np.sin(yaw_prev)) / dt
        self.ang_vel = (self.pose_f[2] - self.prev_pose_f[2]) / dt

        # LPF velocity
        if self.tau_vel > 0:
            alpha_vel = dt / (self.tau_vel + dt)
            self._prev_lin_vel_f = self.lin_vel_f
            self._prev_ang_vel_f = self.ang_vel_f
            self.lin_vel_f = (1 - alpha_vel) * self.lin_vel_f + alpha_vel * self.lin_vel
            self.ang_vel_f = (1 - alpha_vel) * self.ang_vel_f + alpha_vel * self.ang_vel
            vel_for_acc_lin = self.lin_vel_f
            vel_for_acc_ang = self.ang_vel_f
            prev_vel_for_acc_lin = self._prev_lin_vel_f
            prev_vel_for_acc_ang = self._prev_ang_vel_f
        else:
            vel_for_acc_lin = self.lin_vel
            vel_for_acc_ang = self.ang_vel
            prev_vel_for_acc_lin = self._prev_lin_vel
            prev_vel_for_acc_ang = self._prev_ang_vel

        # Diferentiate velocity to get acceleration
        self.prev_lin_acc = self.lin_acc
        self.prev_ang_acc = self.ang_acc
        self.lin_acc = (vel_for_acc_lin - prev_vel_for_acc_lin) / dt
        self.ang_acc = (vel_for_acc_ang - prev_vel_for_acc_ang) / dt

        return()
    
    def get_dist_and_theta(self, agent):
        # Slice pose
        x1, y1, theta1 = self.pose
        x2, y2, theta2 = agent.pose

        # Calculate dist and theta
        d = np.hypot((x2-x1),(y2-y1))
        theta = self.wrap_to_pi(theta2 - theta1)

        return d, theta
        

    def reset(self):
        self.time_meas = None
        self.prev_time_meas = None
        self.pose = None
        self.prev_pose = None
        self.lin_vel = None
        self._prev_lin_vel = 0
        self.ang_vel = None
        self._prev_ang_vel = 0
        self.lin_acc = None  
        self.prev_lin_acc = 0
        self.ang_acc = None
        self.prev_ang_acc = 0
        self.pose_f = None
        self.prev_pose_f = None
        self._yaw_unwrapped = None
        self._prev_yaw_raw = None
        self.lin_vel_f = 0.0
        self.ang_vel_f = 0.0
        self._prev_lin_vel_f = 0.0
        self._prev_ang_vel_f = 0.0

    def wrap_to_pi(self, a: float) -> float:
        # Wrap angle to [-pi, pi]
        return (a + np.pi) % (2 * np.pi) - np.pi
        
        

def camera_setup(width=1280, height=720, fps=100):
    print("\nInitializing camera and detector...")
    # Initialize camera and detector

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # switch to DirectShow
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)    # 1280 x 720
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Latency / buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Lock exposure/focus (DSHOW + this camera usually respect these)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # often 0.25 = manual, 0.75 = auto (driver-dependent)
    # set a short exposure (value is camera/driver-dependent; try negative or small positive)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_GAIN, 0)
    if not cap.isOpened():
        print('Failed to open camera')
        exit()

    return cap

class UDP():
    def __init__(self, IP="10.40.109.62", Freq=60):
        self.JETBOT_IP = IP
        self.PORT = 5005
        self.SEND_HZ = Freq
        self.period = 1.0 / self.SEND_HZ
        self.seq=0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)

        # seq(uint32), t_sent(double), left(float), right(float)
        self.PACK_FMT = "<Idff"
        self.PACK_SIZE = struct.calcsize(self.PACK_FMT)
        print(f"[START] Sending to {self.JETBOT_IP}:{self.PORT} Pack_FMT={self.PACK_FMT}")

    def Send(self, left, right):
        t_sent = time.time()  # wall time so JetBot can compute age
        pkt = struct.pack(self.PACK_FMT, self.seq, t_sent, float(left), float(right))
        self.sock.sendto(pkt, (self.JETBOT_IP, self.PORT))
        self.seq += 1

    def Close(self):
        print("\n[STOP] Sending stop command and exiting...")
        t_sent = time.time()
        pkt = struct.pack(self.PACK_FMT, self.seq, t_sent, 0.0, 0.0)
        self.sock.sendto(pkt, (self.JETBOT_IP, self.PORT))
        self.sock.close()

def rot_to_rpy(R):
    """
    Convert 3x3 rotation matrix R (numpy array) to roll, pitch, yaw
    using ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    Returns (roll, pitch, yaw) in radians.
    Robust to numerical issues and handles gimbal-lock.
    """
    # clip for safety
    r20 = float(R[2, 0])
    r20 = max(-1.0, min(1.0, r20))
    pitch = math.asin(-r20)  # theta

    cos_pitch = math.cos(pitch)
    # if cos_pitch is near zero, we are close to gimbal lock
    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])   # phi
        yaw  = math.atan2(R[1, 0], R[0, 0])   # psi
    else:
        # Gimbal lock: pitch ~= +/- pi/2
        # set roll = 0 and compute yaw from other elements
        roll = 0.0
        # yaw ambiguous; derive from R[0,1], R[1,1]
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw