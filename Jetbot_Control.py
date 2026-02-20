import socket
import struct
import time
import numpy as np
import cv2
import math
import AprilTags 
import Motion_Control
import Jetbot_Setup
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import Data_Visualization

# Directory where this file lives
HERE = Path(__file__).parent

def save_to_pickle(data: dict, filename: str):
    """Save data dictionary to a pickle file in the same folder as this script."""
    path = HERE / filename
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_from_pickle(filename: str):
    """Load data dictionary from a pickle file in the same folder as this script."""
    path = HERE / filename
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
# =====================================================================
# Setup
# =====================================================================
    # Setup camera
    cap = Jetbot_Setup.camera_setup(1280, 720)
    frame_count = 0

    # Testing
    at_count = 0

    collect_data = 1    #0=no collect 1=collect 
    if collect_data:
        # Pre-allocate arrays for data collection (over-allocate for safety)
        max_samples = 7200     # 60Hz => 60(samples/s)*120s = 7200samples
        data_time = np.zeros(max_samples)             # Time [s]
        data_pos = np.zeros((max_samples, 3))         # Jetbot pose [x,y,z] [mm]
        data_lin_vel = np.zeros(max_samples)          # Jetbot lin velocity [mm/s]
        data_ang_vel = np.zeros(max_samples)          # Jetbot ang velocity [rad/s]
        data_lin_acc = np.zeros(max_samples)          # Jetbot lin acceleration [mm/s^2]
        data_ang_acc = np.zeros(max_samples)          # Jetbot ang acceleration [rad/s^2]
        count = 0  # Sample counter

    # AprilTag detector
    detector = AprilTags.AprilTags()
    # TODO: Set the validation tag size in millimeters
    TAG_SIZE = 65  # Update this value
    
    intrinsics = np.load('camera_intrinsics.npy')  # Replace with loaded transformation
    print(f"Intrinsics: {intrinsics}")

    # Load the calibration transformation matrix
    T_cam_to_workspace = np.load('camera_workspace_transform.npy')  # Replace with loaded transformation
    print("\nLoaded camera-to-workspace transformation matrix:")
    print(T_cam_to_workspace)

    # TODO: Setup UDP communication
    UDP = Jetbot_Setup.UDP(Freq=60)
    '''
    ssh jetbot@10.40.109.62
    '''

    # Controllers
    # TODO: tune controllers
    pidv = Motion_Control.PID(0.75,0.3,0) # PID for V
    pidw = Motion_Control.PID(0.5,0.2,0) # PID for w
    # max vel[mm/s], max angvel[rad/s], linmax acc[mm/s^2], send freq, pids
    controller = Motion_Control.control(450, 8, 500, UDP.SEND_HZ, pidv, pidw)       
    # TODO: Controller goals
    # min=0 max = 
    A_GOAL = 100     # mm/s^2
    # min=40 max= 500
    V_GOAL = 0    # mm/s
    # min=0 max=10
    W_GOAL = 0      # rad/s

    # Jetbots
    follower1 = Jetbot_Setup.Jetbot(26,0)   # TagID, 0-follower

    initial_time = time.perf_counter()
# =====================================================================
# MAIN TRACKING LOOP
# =====================================================================
    try:
        while True:
            # Record start time of the loop
            start_time = time.perf_counter()
            frame_count += 1
            follower1.visible = 0   # reset visible
            

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
                if (frame_count % 2) == 0:      
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

                        # Collect Data
                        pose = [float(pos_workspace[0]), float(pos_workspace[1]), float(yaw)]
                        if tag_id == follower1.id:
                            follower1.update_meas(pose, time.perf_counter())
                            follower1.visible = 1

                            if collect_data & follower1.visible:
                                data_time[count] = time.perf_counter() - initial_time        # Time [s]
                                data_pos[count, :] = follower1.pose              # Jetbot pose [x,y,z] [mm]
                                data_lin_vel[count] = follower1.lin_vel          # Jetbot lin velocity [mm/s]
                                data_ang_vel[count] = follower1.ang_vel          # Jetbot ang velocity [rad/s]
                                data_lin_acc[count] = follower1.lin_acc          # Jetbot lin acceleration [mm/s^2]
                                data_ang_acc[count] = follower1.ang_acc          # Jetbot ang acceleration [rad/s^2]
                                count += 1

                        # Draw detection on image
                        detector.draw_tags(color_frame, tag)
                        corners = np.asarray(tag.corners, dtype=np.float32)  # (4,2)

            # -----------------------------------------------------------------
            # STEP 4: CONTROLLER AND COMMUNICATION
            # -----------------------------------------------------------------
            if follower1.visible:
                t_now = time.perf_counter()
                # VW controller:
                # v_cmd, w_cmd = controller.controller_vw([follower1.lin_vel, follower1.ang_vel], [V_GOAL, W_GOAL])
                # AW controller
                v_cmd , w_cmd = controller.controller_aw([follower1.lin_vel, follower1.ang_vel],[A_GOAL, W_GOAL])
                left, right = controller.motor_controller(v_cmd, w_cmd)
                at_count += 1 #TESTING
            else:
                left = right = 0.0

            # Send UDP package
            # left = -0.15
            # right = 0.15
            UDP.Send(left, right)

            if (frame_count % 4) == 0:
                # Show instruction
                cv2.putText(color_frame, "Press 'q' to quit", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Camera', color_frame)

                # print(f"Visible: {follower1.visible}")
                # if follower1.visible:
                #     print(f"vels: {follower1.lin_vel}    {follower1.ang_vel}")
                #     print(f"cmd: {v_cmd}    {w_cmd}")
                #     print(f"left {left} right {right}")


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
        print(f"%Time Tag visible: {100*at_count/frame_count}%")

        if collect_data:
            # Trim unused portions of pre-allocated arrays
            data_time = data_time[:count]
            data_pos = data_pos[:count, :]
            data_lin_vel = data_lin_vel[:count]
            data_ang_vel = data_ang_vel[:count]
            data_lin_acc = data_lin_acc[:count]
            data_ang_acc = data_ang_acc[:count]

            # Save all data to pickle file
            filename='Jetbot_Tracking.pkl'
            # Create dictionary with all collected data and control parameters
            print(f"\nSaving data to {filename}...")
            data_dict = {
            'time': data_time, 
            'pos': data_pos, 
            'lin_vel': data_lin_vel,
            'ang_vel': data_ang_vel,
            'lin_acc': data_lin_acc,
            'ang_acc': data_ang_acc
            }

            # Write dictionary to pickle file
            save_to_pickle(data_dict, filename)
            print("Data saved successfully!")

        print("Done!")
    
def plots():
    plot=Data_Visualization    
    # Load data from pickle files
    data = load_from_pickle('Jetbot_Tracking.pkl')
    
    time = data["time"]
    pose = data["pos"]
    lin_vel = data["lin_vel"]
    ang_vel = data["ang_vel"]
    lin_acc = data["lin_acc"]
    ang_acc = data["ang_acc"]


    plot.plot_xy_trajectory(pose)
    plot.plot_xy_vs_time(time, pose)
    plot.plot_velocities(time, lin_vel, ang_vel, v_goal=None, w_goal=None)
    plot.plot_accelerations(time, lin_acc, ang_acc, lin_acc_des=200,window=40)
    plot.plot_aw(time, lin_acc, ang_vel, lin_acc_des=200, ang_vel_des=0, window=50)


    t_start = 1.0  # seconds
    mask = time >= t_start
    steady_lvel = lin_vel[mask]
    avg_lvel = np.mean(steady_lvel)
    steady_avel = ang_vel[mask]
    avg_avel = np.mean(steady_avel)
    steady_lacc = lin_acc[mask]
    avg_lacc = np.mean(steady_lacc)

    np.set_printoptions(precision=5, suppress=1)    # set print precision and suppression
    print("Average steady lin velocity:", avg_lvel, "mm/s")
    print("Average steady lin acceleration:", avg_lacc, "mm/s^2")
    print("Average steady ang velocity:", avg_avel, "rad/s")


    plt.show()

if __name__ == "__main__":
    main()
    plots()