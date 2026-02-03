import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags
import Simple_Controller 

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
    
    # AprilTag detector
    detector = AprilTags.AprilTags()

    # Controller
    controller = Simple_Controller.control(0.3, 0.5 , 130)    # max vel[mm/s] max angvel[rad/s] deadzone[mm]
    # Latest poses: [x, y, yaw] in workspace, yaw radians
    leader = None
    follower1 = None
    LEADER_ID = 9
    FOLLOWER_ID = 26

    # Follower v and w
    f1_pose_prev = None
    f1_t_prev = None
    f1_v = 0.0
    f1_w = 0.0
    
    # TODO: Camera Intrinsics
    fx = 1014.7877227030419
    fy = 1015.5790339720445
    ppx = 426.0549465833697
    ppy = 269.5657379492221

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

    # seq(uint32), t_sent(double), left(float), right(float)
    PACK_FMT = "<Idff"
    PACK_SIZE = struct.calcsize(PACK_FMT)

    print(f"[START] Sending to {JETBOT_IP}:{PORT} Pack_FMT={PACK_FMT}")
    print("Ctrl+C to quit.\n")

    # =====================================================================
    # MAIN TRACKING LOOP
    # =====================================================================
    while True:
        # Record start time of the loop
        start_time = time.perf_counter()  # Replace with actual time
        # Reset each frame; we’ll set if seen
        leader = None
        follower1 = None

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
        
        # If no tags are detected 
        if len(tags)==0:
            # No tags detected            
            cv2.putText(color_frame, "No tag detected", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)

        else: 
            for tag in tags:
                
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

                    # Collect Data
                    pose = [float(pos_workspace[0]), float(pos_workspace[1]), float(yaw)]
                    if tag_id == LEADER_ID:
                        leader = pose
                    if tag_id == FOLLOWER_ID:
                        follower1 = pose

                    # Draw detection on image
                    detector.draw_tags(color_frame, tag)
                    corners = np.asarray(tag.corners, dtype=np.float32)  # (4,2)

                    # # --- Robust "bottom-left-ish" corner under perspective ---
                    # max_y = float(np.max(corners[:, 1]))
                    # tol = 6.0  # pixels; increase if needed
                    # bottom = corners[corners[:, 1] >= max_y - tol]
                    # bl = bottom[np.argmin(bottom[:, 0])]
                    # x = int(bl[0])
                    # y = int(bl[1] + 5)  # padding below tag
                    # h, w = color_frame.shape[:2]
                    # line_h = 12
                    # num_lines = 1 + 3
                    # x = max(5, min(x, w - 120))                  # 120 is a rough text width clamp
                    # y = max(line_h, min(y, h - num_lines*line_h))
                    # # Font settings
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # id_scale = 0.30
                    # xyz_scale = 0.23
                    # thickness = 1
                    # # --- Draw Tag ID first ---
                    # cv2.putText(color_frame, f"Tag ID: {tag.tag_id}",
                    #             (x, y), font, id_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    # # --- Start xyz BELOW the Tag ID to avoid overlap ---
                    # start_y = y + line_h
                    # labels = ["x", "y", "z"]
                    # for i, lab in enumerate(labels):
                    #     cv2.putText(color_frame, f"{lab}: {pos_workspace[i]:.1f} mm",
                    #                 (x, start_y + i*line_h),
                    #                 font, xyz_scale, (0, 255, 255), thickness, cv2.LINE_AA)
                    
            
        # -----------------------------------------------------------------
        # STEP 4: DISPLAY AND USER INTERACTION
        # -----------------------------------------------------------------
        
        # Controller
        if leader is not None and follower1 is not None:
            v, w = controller.controller(follower1, leader)
            left, right = controller.motor_controller(v, w)
        else:
            left, right = 0.0, 0.0

        # -------------------------------------------------------------
        # Compute follower linear speed (mm/s) and angular vel (rad/s)
        # -------------------------------------------------------------
        
        f1_t_now = time.perf_counter()
        if follower1 is not None:
            # Slice x, y, yaw
            x, y, yaw = follower1

            # Check for first dt
            if f1_pose_prev is not None and f1_t_prev is not None:
                dt = f1_t_now - f1_t_prev   # Calc dt

                # Calc deltas
                dx = x - f1_pose_prev[0]
                dy = y - f1_pose_prev[1]
                dyaw = (yaw - f1_pose_prev[2] + math.pi) % (2*math.pi) - math.pi

                # Calc v and w
                f1_v = np.hypot(dx, dy) / dt
                f1_w = dyaw / dt

            # Update prev
            f1_pose_prev = [x, y, yaw]
            f1_t_prev = f1_t_now

        else: 
            f1_v = f1_w = 0.0
            

        # -------------------------------------------------------------
        # Draw follower velocity readout (bottom-right) - stable
        # - fixed sign column (always + or -)
        # - fixed decimal places
        # - fixed box size (template)
        # -------------------------------------------------------------
        h, w_img = color_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        margin = 10
        gap = 6
        pad = 8  # padding inside the box

        # Fixed-width formatted lines:
        #  +  => always show sign so the sign column never appears/disappears
        #  8.1f / 8.2f => fixed field width + fixed decimals => decimal point stays in same place
        line1 = f"Follower v: {f1_v:+8.1f} mm/s"
        line2 = f"Follower w: {f1_w:+8.2f} rad/s"

        # Fixed "worst case" templates to lock the box size (match sign + decimals)
        tmpl1 = "Follower v: +9999.9 mm/s"
        tmpl2 = "Follower w: +9999.99 rad/s"

        (tw1, th1), _ = cv2.getTextSize(tmpl1, font, scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(tmpl2, font, scale, thickness)
        tw = max(tw1, tw2)
        th = th1 + th2 + gap

        # Anchor box bottom-right using w_img/h
        x = w_img - margin - (tw + 2 * pad)
        y = h     - margin - (th + 2 * pad)

        # Background rectangle
        cv2.rectangle(
            color_frame,
            (x, y),
            (x + tw + 2 * pad, y + th + 2 * pad),
            (0, 0, 0),
            -1
        )

        # Text baselines
        y1 = y + pad + th1
        y2 = y1 + gap + th2

        cv2.putText(color_frame, line1, (x + pad, y1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(color_frame, line2, (x + pad, y2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)




        # Send UDP package
        t_sent = time.time()  # wall time so JetBot can compute age
        pkt = struct.pack(PACK_FMT, seq, t_sent, float(left), float(right))
        sock.sendto(pkt, (JETBOT_IP, PORT))
        seq += 1

        # Show instruction
        cv2.putText(color_frame, "Press 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Camera', color_frame)
        
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[STOP] Sending stop command and exiting...")
            t_sent = time.time()
            pkt = struct.pack(PACK_FMT, seq, t_sent, 0.0, 0.0)
            sock.sendto(pkt, (JETBOT_IP, PORT))

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