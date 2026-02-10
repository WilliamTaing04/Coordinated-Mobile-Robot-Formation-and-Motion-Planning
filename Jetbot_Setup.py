import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags

def camera_setup(width=0, height=0, fps=60):
    print("\nInitializing camera and detector...")
    # Initialize camera and detector
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if width != 0 and height != 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps != 60:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        print('Failed to open camera')
        exit()
    focus = 0
    print(f"Focus: {focus}")
    # cap.set(cv2.CAP_PROP_FOCUS, focus)
    return cap

class UDP():
    def __init__(self, IP="10.40.109.62", Freq=60):
        self.JETBOT_IP = "{IP}"
        self.PORT = 5005
        self.SEND_HZ = Freq
        self.period = 1.0 / self.SEND_HZ
        self.seq=0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)

        # seq(uint32), t_sent(double), left(float), right(float)
        PACK_FMT = "<Idff"
        PACK_SIZE = struct.calcsize(PACK_FMT)
        print(f"[START] Sending to {self.JETBOT_IP}:{self.PORT} Pack_FMT={PACK_FMT}")

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