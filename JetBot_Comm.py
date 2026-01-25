import socket
import struct
import time
import numpy as np
import cv2
import pickle
import AprilTags

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
        
    JETBOT_IP = "172.20.10.6"   # <-- CHANGE THIS
    PORT = 5005
    SEND_HZ = 50
    PACK_FMT = "<IBdfff"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    period = 1.0 / SEND_HZ
    seq = 0

    # =====================================================================
    # MAIN TRACKING LOOP
    # =====================================================================
    while True:
        # Record start time of the loop
        start_time = time.time()  # Replace with actual time

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
        
        # -----------------------------------------------------------------
        # STEP 3: PROCESS DETECTED TAGS
        # -----------------------------------------------------------------
        
        # UDP variables
        valid = 0
        xw = yw = zw = 0.0

        # Check if any tags were detected
        if len(tags) > 0:
            
            # Use the first detected tag
            tag = tags[0]
            
            # Use detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
            # Returns: (rotation_matrix, translation_vector)
            rot_matrix, trans_vector = detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)

            
            # Check if pose estimation was successful
            if rot_matrix is not None and trans_vector is not None:
                
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
                
                # Calculate distance from camera
                distance = np.linalg.norm(pos_workspace)  # Replace with distance

                # Update UDP packet
                valid = 1
                xw, yw, zw = map(float, pos_workspace)

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
            
        else:
            # No tags detected            
            cv2.putText(color_frame, "No tag detected", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
        
        # -----------------------------------------------------------------
        # STEP 4: DISPLAY AND USER INTERACTION
        # -----------------------------------------------------------------
        
        # Send UDP package
        t_sent = time.perf_counter()
        packet = struct.pack(PACK_FMT, seq, valid, t_sent, xw, yw, zw)
        sock.sendto(packet, (JETBOT_IP, PORT))
        seq += 1


        # Show instruction
        cv2.putText(color_frame, "Press 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Calibration Validation', color_frame)
        
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped sending.")
            stop_packet = struct.pack(PACK_FMT, seq, 0, time.perf_counter(), 0.0, 0.0, 0.0)
            sock.sendto(stop_packet, (JETBOT_IP, PORT))
            cap.release()
            cv2.destroyAllWindows()
            sock.close()
            break


        # -----------------------------------------------------------------
        # MAINTAIN FIXED TIMESTEP
        # -----------------------------------------------------------------
        
        # Enforce consistent loop timing
        elapsed = time.time() - start_time
        if elapsed < period:
            time.sleep(period - elapsed)
    


if __name__ == "__main__":
    main()


'''
import socket
import struct
import time

from jetbot import Robot

# =========================
# UDP + PACKET CONFIG
# =========================
PORT = 5005
PACK_FMT = "<IBdfff"  # seq(uint32), valid(uint8), t_sent(float64), x,y,z(float32)
PACK_SIZE = struct.calcsize(PACK_FMT)

# Stop if packets stop arriving
WATCHDOG_SEC = 0.6

# Drop packets that are too old (prevents "chasing history" backlog delay)
MAX_PACKET_AGE_SEC = 0.25  # 250 ms

# Reduce OS buffering (helps prevent large queues)
RCVBUF_BYTES = 1 << 16  # 64 KB

# =========================
# CONTROL CONFIG (1D PID on x)
# =========================
X_SETPOINT_MM = 200.0    # desired x position in mm (requested)

DEADBAND_MM = 15.0       # stop when within +/- this many mm (avoid jitter)

LOOP_HZ = 20.0
DT_TARGET = 1.0 / LOOP_HZ

# PID gains (start conservative; tune)
KP = 0.003
KI = 0.0000
KD = 0.001

# Speed limits (requested max speed = 0.2)
SPEED_MAX = 0.15
SPEED_MIN = 0.10         # minimum speed to overcome friction (raise to 0.12-0.15 if needed)

# Anti-windup clamp (integrator units: mm*s)
I_CLAMP = 300.0

# Flip if direction is reversed
SIGN = 1.0

# =========================
# DEBUG PRINTING
# =========================
PRINT_CTRL_EVERY_SEC = 0.25   # detailed control print
PRINT_UDP_EVERY_SEC = 1.0     # packet-rate print


def clamp(v, lo, hi):
    return max(lo, min(hi, v))



def main():
    robot = Robot()
    robot.stop()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(0.02)  # short timeout so we can drain quickly each tick

    print(f"[START] JetBot UDP PID controller")
    print(f"[START] Listening on 0.0.0.0:{PORT}")
    print(f"[START] PACK_FMT={PACK_FMT} PACK_SIZE={PACK_SIZE} bytes")
    print(f"[START] Watchdog={WATCHDOG_SEC}s  MaxAge={MAX_PACKET_AGE_SEC}s  RCVBUF={RCVBUF_BYTES} bytes")
    print(f"[START] Setpoint x={X_SETPOINT_MM}mm  deadband={DEADBAND_MM}mm  loop={LOOP_HZ}Hz")
    print(f"[START] PID KP={KP} KI={KI} KD={KD}  speed=[{SPEED_MIN},{SPEED_MAX}]  SIGN={SIGN}")
    print("Ctrl+C to stop.\n")

    # Uncomment once to verify motors respond:
    # motor_sanity_test(robot)

    # PID state
    integ = 0.0
    prev_err = 0.0
    prev_t = time.perf_counter()

    # Latest measurement
    last_rx_wall = None           # perf_counter time of last accepted packet
    last_seq = None
    latest_valid = 0
    latest_tsent = 0.0
    latest_x = latest_y = latest_z = 0.0
    latest_age = None
    latest_src = "?"

    # Debug rate stats
    pkt_window = 0
    udp_last_print = time.perf_counter()
    window_start = time.perf_counter()

    ctrl_last_print = 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            # =========================
            # RECEIVE: DRAIN SOCKET, USE NEWEST ONLY
            # =========================
            newest = None
            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    if len(data) >= PACK_SIZE:
                        newest = (data, addr)
                except socket.timeout:
                    break

            if newest is not None:
                data, addr = newest
                seq, valid, t_sent, x, y, z = struct.unpack(PACK_FMT, data[:PACK_SIZE])

                now = time.perf_counter()
                age = now - float(t_sent)

                # Drop stale packets (prevents backlog delay)
                if age <= MAX_PACKET_AGE_SEC:
                    # Optional: drop out-of-order seq
                    if last_seq is None or seq > last_seq:
                        last_seq = seq
                        latest_valid = int(valid)
                        latest_tsent = float(t_sent)
                        latest_x = float(x)
                        latest_y = float(y)
                        latest_z = float(z)
                        latest_age = age
                        latest_src = addr[0]
                        last_rx_wall = now

                        pkt_window += 1

            # =========================
            # UDP RATE PRINT (once/sec)
            # =========================
            now = time.perf_counter()
            if now - udp_last_print >= PRINT_UDP_EVERY_SEC:
                elapsed = now - window_start
                hz = (pkt_window / elapsed) if elapsed > 0 else 0.0
                pkt_window = 0
                window_start = now
                udp_last_print = now

                age_ms = (latest_age * 1000.0) if latest_age is not None else None
                print(
                    f"[UDP] ~{hz:4.1f} pkt/s  last_seq={last_seq}  valid={latest_valid}  "
                    f"x={latest_x:8.1f}  age_ms={age_ms if age_ms is not None else 'None'}  src={latest_src}"
                )

            # =========================
            # SAFETY: WATCHDOG (no recent accepted packet)
            # =========================
            if last_rx_wall is None or (loop_start - last_rx_wall) > WATCHDOG_SEC:
                robot.stop()
                integ = 0.0
                prev_err = 0.0

                if loop_start - ctrl_last_print >= PRINT_CTRL_EVERY_SEC:
                    ctrl_last_print = loop_start
                    age = None if last_rx_wall is None else (loop_start - last_rx_wall)
                    print(f"[STOP] WATCHDOG: no fresh UDP accepted. last_fresh_age={age}")

                time.sleep(max(0.0, DT_TARGET - (time.perf_counter() - loop_start)))
                continue

            # =========================
            # SAFETY: NO TAG
            # =========================
            if latest_valid == 0:
                robot.stop()
                integ = 0.0
                prev_err = 0.0

                if loop_start - ctrl_last_print >= PRINT_CTRL_EVERY_SEC:
                    ctrl_last_print = loop_start
                    age_ms = (latest_age * 1000.0) if latest_age is not None else None
                    print(f"[STOP] NO TAG: valid=0 (fresh UDP). x={latest_x:.1f} age_ms={age_ms}")

                time.sleep(max(0.0, DT_TARGET - (time.perf_counter() - loop_start)))
                continue

            # =========================
            # PID CONTROL (1D ON x)
            # =========================
            t = time.perf_counter()
            dt = t - prev_t
            if dt <= 1e-6:
                dt = DT_TARGET
            prev_t = t

            # error = setpoint - measurement
            err = SIGN * (X_SETPOINT_MM - latest_x)

            if abs(err) <= DEADBAND_MM:
                robot.stop()
                integ *= 0.8
                prev_err = err
                u = 0.0
                speed = 0.0
                direction = "STOP(deadband)"
                derr = 0.0
                p_term = KP * err
                i_term = KI * integ
                d_term = 0.0
            else:
                # integral (anti-windup clamp)
                integ += err * dt
                integ = clamp(integ, -I_CLAMP, I_CLAMP)

                # derivative
                derr = (err - prev_err) / dt
                prev_err = err

                # PID terms
                p_term = KP * err
                i_term = KI * integ
                d_term = KD * derr
                u = p_term + i_term + d_term

                # motor speed command
                speed = clamp(abs(u), SPEED_MIN, SPEED_MAX)

                if u > 0:
                    robot.forward(speed)
                    direction = "FORWARD"
                else:
                    robot.backward(speed)
                    direction = "BACKWARD"

            # =========================
            # VERBOSE CONTROL PRINT
            # =========================
            if loop_start - ctrl_last_print >= PRINT_CTRL_EVERY_SEC:
                ctrl_last_print = loop_start
                age_ms = (latest_age * 1000.0) if latest_age is not None else None
                print(
                    f"[CTRL] x={latest_x:8.1f} sp={X_SETPOINT_MM:8.1f} err={err:8.1f}  "
                    f"P={p_term: .4f} I={i_term: .4f} D={d_term: .4f}  u={u: .4f}  "
                    f"{direction:14s} speed={speed: .3f}  age_ms={age_ms}"
                )

            # =========================
            # MAINTAIN LOOP RATE
            # =========================
            time.sleep(max(0.0, DT_TARGET - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C pressed.")
    finally:
        robot.stop()
        sock.close()
        print("[CLEANUP] robot stopped, socket closed.")


if __name__ == "__main__":
    main()
'''
