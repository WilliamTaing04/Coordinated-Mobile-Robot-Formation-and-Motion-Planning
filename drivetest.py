import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags
import Simple_Controller 
import VW_Controller
import Jetbot_Setup

def main():
# =====================================================================
# Setup
# =====================================================================
    # Setup camera
    cap = Jetbot_Setup.camera_setup(1280, 720, 60)

    # AprilTag detector
    detector = AprilTags.AprilTags()
    # TODO: Set the validation tag size in millimeters
    TAG_SIZE = 65  # Update this value
    
    # TODO: Update Camera Intrinsics
    fx = 1014.7877227030419
    fy = 1015.5790339720445
    ppx = 426.0549465833697
    ppy = 269.5657379492221
    # intrinsics = np.load('camera_intrinsics.npy')  # Replace with loaded transformation

    intrinsics = np.array([
                [fx, 0, ppx],
                [0, fy, ppy],
                [0, 0, 1]])
    print(f"Intrinsics: {intrinsics}")

    # Load the calibration transformation matrix
    T_cam_to_workspace = np.load('camera_workspace_transform.npy')  # Replace with loaded transformation
    print("\nLoaded camera-to-workspace transformation matrix:")
    print(T_cam_to_workspace)

    # Setup UDP communication
    UDP = Jetbot_Setup.UDP()

    # Controllers
    pidv = VW_Controller.PID(1,0,0) # PID for V
    pidw = VW_Controller.PID(2,0,0) # PID for w
    controller = VW_Controller.control(300, 1, UDP.SEND_HZ, pidv, pidw)   # max vel[mm/s], max angvel[rad/s], send freq, pids
    
    # Controller goals (make these easy to tune)
    V_GOAL = 100.0       # mm/s example
    W_GOAL = np.pi/2     # rad/s example

    # Latest poses: [x, y, yaw] in workspace, yaw radians
    leader = None
    follower1 = None
    LEADER_ID = 9
    FOLLOWER_ID = 26

    # Follower v and w
    f1_pose_prev = None
    f1_t_prev = None
    
    # Velocity measurement hold parameters 
    f1_v_meas = 0.0
    f1_w_meas = 0.0

# =====================================================================
# MAIN TRACKING LOOP
# =====================================================================
    try:
        while True:
            # Record start time of the loop
            start_time = time.perf_counter()

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
                        roll, pitch, yaw = Jetbot_Setup.rot_to_rpy(rot_workspace)
                        
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

            # -----------------------------------------------------------------
            # STEP 4: CONTROLLER AND COMMUNICATION
            # -----------------------------------------------------------------
            # TODO: create controller
            left = right = None

            if follower1 is not None:
                t_now = time.perf_counter()

                

            # Send UDP package
            UDP.Send(left, right)

            # Show instruction
            cv2.putText(color_frame, "Press 'q' to quit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Camera', color_frame)
            key = cv2.waitKey(1)

            # Check for quit key press ('q' or ESC)
            if key & 0xFF == ord('q') or key == 27:
                print("\nQuitting...")
                break

            # -----------------------------------------------------------------
            # MAINTAIN FIXED TIMESTEP
            # -----------------------------------------------------------------
            elapsed = time.perf_counter() - start_time
            if elapsed < UDP.period:
                time.sleep(UDP.period - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # =====================================================================
        # CLEANUP
        # =====================================================================
        print("\n[STOP] Sending stop command and exiting...")
        UDP.Close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")
    

if __name__ == "__main__":
    main()