import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags

class Jetbot():
    def __init__(self, id, role=0):
        self.id = id
        self.role = role                # 0-follower 1-leader
        self.visible = 0                # 0-not seen 1-seen
        self.time_meas = None           # time of measurement
        self.prev_time_meas = None      # time of previous measurement
        self.pose = None                # [x, y, theta]
        self.prev_pose = None           # [x, y, theta]
        self.lin_vel = None             # mm/s
        self._prev_lin_vel = 0          # mm/s
        self.ang_vel = None             # rad/s
        self._prev_ang_vel = 0          # rad/s
        self.lin_acc = None             # mm/s^2
        self.prev_lin_acc = 0           # mm/s^2
        self.ang_acc = None             # rad/s^2
        self.prev_ang_acc = 0           # rad/s^2

    def update_meas(self, pose, time_meas):
        pose = np.asarray(pose, dtype=float).copy()
        time_meas = float(time_meas)
        # Check for first update
        if self.pose is None or self.time_meas is None:
            self.pose = pose
            self.prev_pose = pose.copy()
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
            self.prev_time_meas = self.time_meas
            return
        

        # Update pose
        self.prev_pose = self.pose
        self.pose = pose
        
        # Update velocities
        self._prev_lin_vel = self.lin_vel
        self._prev_ang_vel = self.ang_vel
        dx = (self.pose[0] - self.prev_pose[0])
        dy = (self.pose[1] - self.prev_pose[1])
        self.lin_vel = v_forward = (dx*np.cos(self.prev_pose[2]) + dy*np.sin(self.prev_pose[2])) / dt
        self.ang_vel = self.wrap_to_pi((self.pose[2] - self.prev_pose[2])) / dt

        # Update accelerations
        self.prev_lin_acc = self.lin_acc
        self.prev_ang_acc = self.ang_acc
        self.lin_acc = (self.lin_vel - self._prev_lin_vel) / dt
        self.ang_acc = (self.ang_vel - self._prev_ang_vel) / dt

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