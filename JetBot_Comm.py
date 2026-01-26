import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags

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

def main():
    """
    Main tracking routine
    """
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    print("="*60)
    print("Workspace AprilTag Tracking")
    print("="*60)
    
    print("\nInitializing camera and detector...")
    # Initialize camera and detector
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

    # TODO: Update camera settings
    focus = 535
    print(f"Focus: {focus}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)

    if not cap.isOpened():
        print('Failed to open camera')
        exit()
    
    detector = AprilTags.AprilTags()
    
    # TODO: Camera Intrinsics
    # fx = 487.42056093
    # fy = 487.42053388
    # ppx = 317.3216121
    # ppy = 248.73120265
    fx = 490.00332243
    fy = 489.5556459
    ppx = 315.8040739
    ppy = 268.93739803

    intrinsics = np.array([
                [fx, 0, ppx],
                [0, fy, ppy],
                [0, 0, 1]])
    print(f"Intrinsics: {intrinsics}")
    
    # Load the calibration transformation matrix
    T_cam_to_workspace = np.load('camera_workspace_transform.npy')  # Replace with loaded transformation
    print("\nLoaded camera-to-workspace transformation matrix:")
    print(T_cam_to_workspace)
    
    # TODO: Set the validation tag size in millimeters
    # IMPORTANT: Measure your validation tag!
    TAG_SIZE = 65  # Update this value

    print(f"\nValidation tag size: {TAG_SIZE} mm")
    print("\n" + "="*60)
    print("Starting tracking...")
    print("Move the tag around the workspace")
    print("Press 'q' to quit")
    print("="*60 + "\n")
        
    # TODO: Change UDP settings
    JETBOT_IP = "172.20.10.6"
    PORT = 5005
    SEND_HZ = 50
    period = 1.0 / SEND_HZ
    seq=0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)

    # Header: seq (I), t_sent (d), n_tags (B)
    HDR_FMT = "<IdB"
    HDR_SZ = struct.calcsize(HDR_FMT)

    # Per-tag: tag_id (I), x,y,z (f f f), roll,pitch,yaw (f f f)
    TAG_FMT = "<Iffffff"
    TAG_SZ = struct.calcsize(TAG_FMT)

    MAX_TAGS = 10  # safety limit

    print(f"[START] Sending to {JETBOT_IP}:{PORT}  HDR_FMT={HDR_FMT} TAG_FMT={TAG_FMT}")
    print("Ctrl+C to quit.\n")


    # =====================================================================
    # MAIN TRACKING LOOP
    # =====================================================================
    while True:
        # Record start time of the loop
        start_time = time.perf_counter()  # Replace with actual time

        # -----------------------------------------------------------------
        # STEP 1: CAPTURE FRAME
        # -----------------------------------------------------------------
        
        # Get camera frame
        ret, color_frame = cap.read()
        
        if not ret:
            continue
        
        # -----------------------------------------------------------------
        # STEP 2: DETECT APRILTAGS
        # -----------------------------------------------------------------
        
        # Use detector.detect_tags(color_frame)
        tags = detector.detect_tags(color_frame)  # Replace with detected tags
        tag_packet = [] # Created array for packet of tags

        
        # -----------------------------------------------------------------
        # STEP 3: PROCESS DETECTED TAGS
        # -----------------------------------------------------------------
        
        # If no tags are detected 
        if len(tags)==0:
            # No tags detected            
            cv2.putText(color_frame, "No tag detected", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)

        else: 
            for tag in tags[:MAX_TAGS]:
                
                # Use detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                rot_matrix, trans_vector = detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                
                # Check if pose estimation was successful
                if rot_matrix is not None and trans_vector is not None:
                    
                    # Get tag id
                    tag_id = tag.tag_id
                    
                    # Extract position in camera frame (already in mm)
                    # Flatten trans_vector to get a 1D array of shape (3,)
                    pos_camera = trans_vector.flatten()  # Replace with position array

                    # Create full 4x4 pose transformation from tag to camera
                    T_tag_to_cam = np.eye(4)
                    T_tag_to_cam[:3, :3] = rot_matrix  # Replace with rotation matrix
                    T_tag_to_cam[:3, 3] = trans_vector.reshape((3,))  # Replace with translation vector
                    
                    # Transform full pose to workspace frame
                    # Multiply T_cam_to_workspace @ T_tag_to_cam
                    T_tag_to_workspace = T_cam_to_workspace @ T_tag_to_cam  # Replace with homogeneous coordinates (4x4)
                    
                    # Extract position and orientation in workspace frame
                    pos_workspace = T_tag_to_workspace[:3, 3]
                    rot_workspace = T_tag_to_workspace[:3, :3]

                    # convert rotation matrix to rpy (radians)
                    roll, pitch, yaw = rot_to_rpy(rot_workspace)
                    
                    # # Calculate distance from camera
                    # distance = np.linalg.norm(pos_workspace)  # Replace with distance

                    # save for UDP
                    tag_packet.append(
                        (
                            int(tag.tag_id),
                            float(pos_workspace[0]),
                            float(pos_workspace[1]),
                            float(pos_workspace[2]),
                            float(roll),
                            float(pitch),
                            float(yaw),
                        )
                    )

                    # Draw detection on image
                    detector.draw_tags(color_frame, tag)
                    corners = np.asarray(tag.corners, dtype=np.float32)  # (4,2)

                    # --- Robust "bottom-left-ish" corner under perspective ---
                    max_y = float(np.max(corners[:, 1]))
                    tol = 6.0  # pixels; increase if needed
                    bottom = corners[corners[:, 1] >= max_y - tol]
                    bl = bottom[np.argmin(bottom[:, 0])]
                    x = int(bl[0])
                    y = int(bl[1] + 5)  # padding below tag
                    h, w = color_frame.shape[:2]
                    line_h = 12
                    num_lines = 1 + 3
                    x = max(5, min(x, w - 120))                  # 120 is a rough text width clamp
                    y = max(line_h, min(y, h - num_lines*line_h))
                    # Font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    id_scale = 0.30
                    xyz_scale = 0.23
                    thickness = 1
                    # --- Draw Tag ID first ---
                    cv2.putText(color_frame, f"Tag ID: {tag.tag_id}",
                                (x, y), font, id_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    # --- Start xyz BELOW the Tag ID to avoid overlap ---
                    start_y = y + line_h
                    labels = ["x", "y", "z"]
                    for i, lab in enumerate(labels):
                        cv2.putText(color_frame, f"{lab}: {pos_workspace[i]:.1f} mm",
                                    (x, start_y + i*line_h),
                                    font, xyz_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                    
            
            
        
        # -----------------------------------------------------------------
        # STEP 4: DISPLAY AND USER INTERACTION
        # -----------------------------------------------------------------
        
        # Send UDP package
        t_sent = time.perf_counter()
        n = min(len(tag_packet), 255)

        buf = bytearray()
        buf += struct.pack(HDR_FMT, seq, t_sent, n)
        for (tag_id, x, y, z, roll, pitch, yaw) in tag_packet[:n]:
            buf += struct.pack(TAG_FMT, tag_id, x, y, z, roll, pitch, yaw)

        sock.sendto(buf, (JETBOT_IP, PORT))
        seq += 1


        # Show instruction
        cv2.putText(color_frame, "Press 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Camera', color_frame)
        
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped sending.")
            # send stop packet: n_tags=0
            t_stop = time.perf_counter()
            stop_buf = struct.pack(HDR_FMT, seq, t_stop, 0)
            sock.sendto(stop_buf, (JETBOT_IP, PORT))

            cap.release()
            cv2.destroyAllWindows()
            sock.close()
            break


        # -----------------------------------------------------------------
        # MAINTAIN FIXED TIMESTEP
        # -----------------------------------------------------------------
        
        # Enforce consistent loop timing
        elapsed = time.perf_counter() - start_time
        if elapsed < period:
            time.sleep(period - elapsed)
    


if __name__ == "__main__":
    main()


'''
"""
JetBot UDP receiver for multi-AprilTag packets:
Header:  <IdB   => seq(uint32), t_sent(float64), n_tags(uint8)
Per tag: <Iffffff => tag_id(uint32), x,y,z(float32), roll,pitch,yaw(float32)

Features:
- drains socket each loop and uses ONLY the newest packet (prevents backlog delay)
- drops stale packets by sender timestamp (t_sent) (prevents chasing history)
- watchdog: prints + indicates stop condition if no fresh packet arrives

Run on JetBot:
  python3 jetbot_udp_rx_print.py
"""

import socket
import struct
import time

# =========================
# NETWORK + PACKET CONFIG
# =========================
PORT = 5005

HDR_FMT = "<IdB"         # seq, t_sent, n_tags
TAG_FMT = "<Iffffff"     # tag_id, x,y,z, roll,pitch,yaw

HDR_SZ = struct.calcsize(HDR_FMT)
TAG_SZ = struct.calcsize(TAG_FMT)

# =========================
# SAFETY / LATENCY CONTROL
# =========================
WATCHDOG_SEC = 0.6          # if no FRESH packet in this time -> watchdog triggers
MAX_PACKET_AGE_SEC = 0.25   # drop packets older than this (based on t_sent)
RCVBUF_BYTES = 1 << 16      # 64KB receive buffer (smallish to reduce queueing)

# =========================
# PRINT SETTINGS
# =========================
PRINT_EVERY_SEC = 0.2        # how often to print a full summary (5 Hz)
PRINT_DROPS_EVERY_SEC = 1.0  # how often to print drop stats


def unpack_packet(data: bytes):
    """Return (seq, t_sent, tags_list) or None if malformed."""
    if len(data) < HDR_SZ:
        return None
    seq, t_sent, n = struct.unpack_from(HDR_FMT, data, 0)
    n = int(n)

    needed = HDR_SZ + n * TAG_SZ
    if len(data) < needed:
        return None

    tags = []
    off = HDR_SZ
    for _ in range(n):
        tag_id, x, y, z, roll, pitch, yaw = struct.unpack_from(TAG_FMT, data, off)
        tags.append((int(tag_id), float(x), float(y), float(z), float(roll), float(pitch), float(yaw)))
        off += TAG_SZ

    return int(seq), float(t_sent), tags


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(0.02)  # short timeout so loop can run fast and drain queue

    print("[START] JetBot UDP AprilTag receiver (print-only)")
    print(f"[START] Listening on 0.0.0.0:{PORT}")
    print(f"[START] HDR_FMT={HDR_FMT} ({HDR_SZ} bytes)  TAG_FMT={TAG_FMT} ({TAG_SZ} bytes)")
    print(f"[START] WATCHDOG_SEC={WATCHDOG_SEC}  MAX_PACKET_AGE_SEC={MAX_PACKET_AGE_SEC}  RCVBUF_BYTES={RCVBUF_BYTES}")
    print("Ctrl+C to stop.\n")

    last_fresh_rx = None          # perf_counter time we last accepted a fresh packet
    last_seq = None

    # Latest accepted packet
    latest_age = None
    latest_src = None
    latest_seq = None
    latest_n = 0
    latest_tags = []

    # Stats
    drops_stale = 0
    drops_malformed = 0
    drops_out_of_order = 0

    last_print = time.perf_counter()
    last_drop_print = time.perf_counter()

    try:
        while True:
            loop_now = time.perf_counter()

            # =========================
            # RECEIVE: DRAIN SOCKET, KEEP NEWEST ONLY
            # =========================
            newest = None
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    newest = (data, addr)  # keep overwriting -> newest packet wins
                except socket.timeout:
                    break

            if newest is not None:
                data, addr = newest
                parsed = unpack_packet(data)

                if parsed is None:
                    drops_malformed += 1
                else:
                    seq, t_sent, tags = parsed
                    now = time.perf_counter()
                    age = now - t_sent

                    # Drop stale packets (kills backlog delay)
                    if age > MAX_PACKET_AGE_SEC:
                        drops_stale += 1
                    else:
                        # Drop out-of-order seq (optional but helps)
                        if last_seq is not None and seq <= last_seq:
                            drops_out_of_order += 1
                        else:
                            last_seq = seq
                            last_fresh_rx = now

                            latest_src = addr[0]
                            latest_seq = seq
                            latest_age = age
                            latest_tags = tags
                            latest_n = len(tags)

            # =========================
            # WATCHDOG
            # =========================
            watchdog = False
            if last_fresh_rx is None or (loop_now - last_fresh_rx) > WATCHDOG_SEC:
                watchdog = True

            # =========================
            # PRINT STATUS
            # =========================
            if (loop_now - last_print) >= PRINT_EVERY_SEC:
                last_print = loop_now

                if watchdog:
                    since = None if last_fresh_rx is None else (loop_now - last_fresh_rx)
                    print(f"[WATCHDOG] No fresh packets. last_fresh_age={since}")
                else:
                    age_ms = latest_age * 1000.0 if latest_age is not None else None
                    print(f"\n[PKT] src={latest_src} seq={latest_seq} n_tags={latest_n} age_ms={age_ms:.1f}")

                    for (tag_id, x, y, z, roll, pitch, yaw) in latest_tags:
                        print(
                            f"  tag_id={tag_id:3d}  "
                            f"xyz=({x:8.1f},{y:8.1f},{z:8.1f}) mm   "
                            f"rpy=({roll: .3f},{pitch: .3f},{yaw: .3f}) rad"
                        )

            if (loop_now - last_drop_print) >= PRINT_DROPS_EVERY_SEC:
                last_drop_print = loop_now
                print(f"[STATS] drops: stale={drops_stale} malformed={drops_malformed} out_of_order={drops_out_of_order}")

            # tiny sleep to reduce CPU
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C pressed.")
    finally:
        sock.close()
        print("[CLEANUP] socket closed.")


if __name__ == "__main__":
    main()
    '''
