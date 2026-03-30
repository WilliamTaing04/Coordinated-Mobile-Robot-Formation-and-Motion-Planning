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
import Data_Visualization as plot
import farzan_vishrut_algorithm

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
    dt_break = 0
    np.set_printoptions(precision=4, suppress=True)

    # Settings
    collect_data = 1    # 0=no collect 1=collect
    control_freq = 30   # Hz
    TAG_SIZE = 65       # mm

    # AprilTag detector
    detector = AprilTags.AprilTags()
    intrinsics = np.load('camera_intrinsics.npy')  # Replace with loaded transformation
    print(f"Intrinsics: {intrinsics}")

    # Load the calibration transformation matrix
    T_cam_to_workspace = np.load('camera_workspace_transform.npy')  # Replace with loaded transformation
    print(f"Transformation Matrix: {T_cam_to_workspace}")

    # UDP communication
    UDP = Jetbot_Setup.UDP(Freq=control_freq)
    '''
follower1:
ssh jetbot@10.40.109.62
follower2:
ssh jetbot@10.40.101.192
follower3:
ssh jetbot@10.40.122.94
follower4:
ssh jetbot@10.40.122.89

cd ~/jetbot
python3 -m jetbot.control_reciever
    '''

    # Controllers
    pidvL = Motion_Control.PID(0.5,0.1,0) # PID for v
    pidwL = Motion_Control.PID(0.5,0.1,0) # PID for w
    pidv1 = Motion_Control.PID(0.5,0.1,0) # PID for v
    pidw1 = Motion_Control.PID(0.5,0.1,0) # PID for w
    pidv2 = Motion_Control.PID(0.5,0.1,0) # PID for v
    pidw2 = Motion_Control.PID(0.5,0.1,0) # PID for w
    pidv3 = Motion_Control.PID(0.5,0.1,0) # PID for v
    pidw3 = Motion_Control.PID(0.5,0.1,0) # PID for w
    pidv4 = Motion_Control.PID(0.5,0.1,0) # PID for v
    pidw4 = Motion_Control.PID(0.5,0.1,0) # PID for w
    # max vel[mm/s], max angvel[rad/s], linmax acc[mm/s^2], send freq, pids
    controllerL = Motion_Control.control(500, 8, 500, control_freq, pidvL, pidwL, alpha=0.75)
    controller1 = Motion_Control.control(500, 8, 500, control_freq, pidv1, pidw1, alpha=0.75)
    controller2 = Motion_Control.control(500, 8, 500, control_freq, pidv2, pidw2, alpha=0.75)
    controller3 = Motion_Control.control(500, 8, 500, control_freq, pidv3, pidw3, alpha=0.75)
    controller4 = Motion_Control.control(500, 8, 500, control_freq, pidv4, pidw4, alpha=0.75)


    # # Controller goals
    # # min=0 max =400
    # U_GOAL = 0     # mm/s^2
    # # min=40 max=500
    # V_GOAL = 0    # mm/s
    # # min=0 max=10
    # W_GOAL = 0      # rad/s

    # Jetbots
    leader = Jetbot_Setup.Jetbot(9,"bad",controllerL,role=1,tau_pose=0.2,tau_vel=0.25)
    follower1 = Jetbot_Setup.Jetbot(26,"10.40.109.62",controller1,role=0,tau_pose=0.1,tau_vel=0.1)   # TagID, 0-follower
    follower2 = Jetbot_Setup.Jetbot(9992,"10.40.101.192",controller2,role=0,tau_pose=0.1,tau_vel=0.1)   # TagID, 0-follower
    follower3 = Jetbot_Setup.Jetbot(9993,"10.40.122.94",controller3,role=0,tau_pose=0.1,tau_vel=0.1)   # TagID, 0-follower
    follower4 = Jetbot_Setup.Jetbot(9994,"10.40.122.89",controller4,role=0,tau_pose=0.1,tau_vel=0.1)   # TagID, 0-follower
    
    agent1 = farzan_vishrut_algorithm.Agent() #for farzan_vishrut_algorithm
    
    jetbot_array = [leader, follower1]

    if collect_data:
        # Pre-allocate arrays for data collection (over-allocate for safety)
        num_bots = len(jetbot_array)
        max_samples = 7200                              
        data_time = np.zeros(max_samples)                         # Time [s]
        data_pos   = np.full((max_samples, num_bots, 3), np.nan)  # Jetbot pose [x,y,theta] [mm][rad]
        data_pos_f = np.full((max_samples, num_bots, 3), np.nan)  # Jetbot pose [x,y,theta] [mm][rad] (filtered)
        data_lin_vel = np.zeros((max_samples, num_bots))          # Jetbot lin velocity [mm/s]
        data_ang_vel = np.zeros((max_samples, num_bots))          # Jetbot ang velocity [rad/s]
        data_lin_vel_f = np.zeros((max_samples, num_bots))        # Jetbot lin velocity [mm/s] (filtered)
        data_ang_vel_f = np.zeros((max_samples, num_bots))        # Jetbot ang velocity [rad/s] (filtered)
        data_lin_acc = np.zeros((max_samples, num_bots))          # Jetbot lin acceleration [mm/s^2]
        data_ang_acc = np.zeros((max_samples, num_bots))          # Jetbot ang acceleration [rad/s^2]
        data_lin_acc_des = np.zeros((max_samples, num_bots))      # Jetbot lin acceleration [mm/s^2] (desired)
        data_ang_vel_des = np.zeros((max_samples, num_bots))      # Jetbot ang velocity [rad/s] (desired)
        count = 0  # Sample counter

    initial_time = time.perf_counter()
# =====================================================================
# MAIN TRACKING LOOP
# =====================================================================
    try:
        while True:
            # Record start time of the loop
            start_time = time.perf_counter()
            frame_count += 1
            for jetbot in jetbot_array:
                jetbot.visible = 0   # reset visible 

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
                if (frame_count % 4) == 0:      
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
                        pos_camera = trans_vector.flatten()  # Replace with position array

                        # Create full 4x4 pose transformation from tag to camera
                        T_tag_to_cam = np.eye(4)
                        T_tag_to_cam[:3, :3] = rot_matrix  # Replace with rotation matrix
                        T_tag_to_cam[:3, 3] = trans_vector.reshape((3,))  # Replace with translation vector
                        
                        # Transform full pose to workspace frame
                        T_tag_to_workspace = T_cam_to_workspace @ T_tag_to_cam  # Replace with homogeneous coordinates (4x4)
                        
                        # Extract position and orientation in workspace frame
                        pos_workspace = T_tag_to_workspace[:3, 3]
                        rot_workspace = T_tag_to_workspace[:3, :3]

                        # convert rotation matrix to rpy (radians)
                        roll, pitch, yaw = Jetbot_Setup.rot_to_rpy(rot_workspace)

                        # Get pose for data collection
                        pose = [float(pos_workspace[0]), float(pos_workspace[1]), float(yaw)]
                        
                        # TODO: multi agent
                        # for jetbot in jetbot_array:
                        #     if tag_id == jetbot.id:
                        #         t_meas = time.perf_counter()
                        #         jetbot.update_meas(pose, t_meas)
                        #         jetbot.visible = 1
                        #         d, v, theta = jetbot.get_dist_theta(leader)
                        #         updated = np.array([d/1000, v/1000, theta])
                        #         agent1.update_self_state(updated,updated)
                                    
                        if tag_id == leader.id:
                            t_meas = time.perf_counter()
                            leader.update_meas(pose, t_meas)
                            leader.visible = 1

                        if tag_id == follower1.id:
                            t_meas = time.perf_counter()
                            follower1.update_meas(pose, t_meas)
                            d, v, theta = follower1.get_dist_theta(leader) # [mm, mm/s, radians]
                            updated = np.array([d/1000, v/1000, theta]) # mm to m [m, m/s, radians]
                            #print("distance:", updated[0])
                            agent1.update_self_state(updated,updated)

                            observed = agent1.observed
                            if(abs(observed[0,0]/observed[0,3] - d) < 0.05):
                                gain = 0.01
                            else:
                                gain = 1

                            follower1.visible = 1

                            # if collect_data:
                            #     data_time[count] = t_meas - initial_time         # Time [s]
                            #     data_pos[count, :] = follower1.pose              # Jetbot pose [x,y,theta] [mm][rad]
                            #     data_pos_f[count, :] = follower1.pose_f          # Jetbot pose [x,y,theta] [mm][rad] (filtered)
                            #     data_lin_vel[count] = follower1.lin_vel          # Jetbot lin velocity [mm/s]
                            #     data_ang_vel[count] = follower1.ang_vel          # Jetbot ang velocity [rad/s]
                            #     data_lin_vel_f[count] = follower1.lin_vel_f      # Jetbot lin velocity [mm/s] (filtered)
                            #     data_ang_vel_f[count] = follower1.ang_vel_f      # Jetbot ang velocity [rad/s] (filtered)
                            #     data_lin_acc[count] = follower1.lin_acc          # Jetbot lin acceleration [mm/s^2]
                            #     data_ang_acc[count] = follower1.ang_acc          # Jetbot ang acceleration [rad/s^2]
                            #     count += 1
                            # Draw detection on image

                    if (frame_count % 4) == 0:
                        detector.draw_tags(color_frame, tag)
                        corners = np.asarray(tag.corners, dtype=np.float32)  # (4,2)

            if collect_data and count < max_samples:
                t_meas = time.perf_counter()
                data_time[count] = t_meas - initial_time                 # Time [s]
                for i, jetbot in enumerate(jetbot_array):
                    if jetbot.visible:
                        data_pos[count, i, :] = jetbot.pose              # Jetbot pose [x,y,theta] [mm][rad]
                        data_pos_f[count, i, :] = jetbot.pose_f          # Jetbot pose [x,y,theta] [mm][rad] (filtered)
                        data_lin_vel[count, i] = jetbot.lin_vel          # Jetbot lin velocity [mm/s]
                        data_ang_vel[count, i] = jetbot.ang_vel          # Jetbot ang velocity [rad/s]
                        data_lin_vel_f[count, i] = jetbot.lin_vel_f      # Jetbot lin velocity [mm/s] (filtered)
                        data_ang_vel_f[count, i] = jetbot.ang_vel_f      # Jetbot ang velocity [rad/s] (filtered)
                        data_lin_acc[count, i] = jetbot.lin_acc          # Jetbot lin acceleration [mm/s^2]
                        data_ang_acc[count, i] = jetbot.ang_acc          # Jetbot ang acceleration [rad/s^2]
                count += 1

                        
                

            # -----------------------------------------------------------------
            # STEP 4: CONTROLLER AND COMMUNICATION
            # -----------------------------------------------------------------
            # TODO: mutli agent
            for i, jetbot in enumerate(jetbot_array):
                if jetbot.visible and jetbot.role==0:
                    agent1.RK4_step()   # TODO: change this
                    U_GOAL, W_GOAL = agent1.getuw()
                    data_lin_acc_des[count-1, i] = U_GOAL * 1000  # m/s^2 -> mm/s^2 if that's what U_GOAL is
                    data_ang_vel_des[count-1, i] = W_GOAL
                    t_now = time.perf_counter()
                    # VW controller:
                    # v_cmd, w_cmd = controller.controller_vw([follower1.lin_vel, follower1.ang_vel], [V_GOAL, W_GOAL])
                    # UW controller
                    v_cmd , w_cmd = jetbot.controller.controller_uw([jetbot.lin_vel, jetbot.ang_vel],[gain*U_GOAL*1000, gain*W_GOAL])
                    left, right = jetbot.controller.motor_controller(v_cmd, w_cmd)

                else:
                    left = right = 0.0
                    
                # Send UDP package
                # left = 0.2
                # right = -0.2
                if jetbot.role==0:
                    UDP.Send(jetbot.IP, left, right)
            at_count += 1 #TESTING

            # TODO: uncomment for normal use
            # if follower1.visible:
            #     agent1.RK4_step()
            #     U_GOAL, W_GOAL = agent1.getuw()
            #     data_lin_acc_des[count-1, 1] = U_GOAL * 1000  # m/s^2 -> mm/s^2 if that's what U_GOAL is
            #     data_ang_vel_des[count-1, 1] = W_GOAL
            #     t_now = time.perf_counter()
            #     # VW controller:
            #     # v_cmd, w_cmd = controller.controller_vw([follower1.lin_vel, follower1.ang_vel], [V_GOAL, W_GOAL])
            #     # UW controller
            #     v_cmd , w_cmd = controller.controller_uw([follower1.lin_vel, follower1.ang_vel],[U_GOAL, W_GOAL])
            #     left, right = controller.motor_controller(v_cmd, w_cmd)

            #     at_count += 1 #TESTING
            # else:
            #     left = right = 0.0

            # Send UDP package
            #left = 0
            #right = 0
            #if(left < 0) and (left < 0):
             #   left = 0;
              #  right = 0;
            # UDP.Send(left, right)

            # Reduce display
            if (frame_count % 4) == 0:
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
            if elapsed < 1/control_freq:
                time.sleep(1/control_freq - elapsed)
            else:
                dt_break += 1
                # print("Loop period exceeded")
                # break
                pass

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
        for jetbot in jetbot_array:
            if jetbot.role==0:
                UDP.Close(jetbot.IP)
        UDP.Shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print(f"%Time Tag visible: {100*at_count/frame_count}%")
        print(f"%Time DT exceeded: {100*dt_break/frame_count}% {dt_break}")

        if collect_data:
            # Trim unused portions of pre-allocated arrays
            data_time = data_time[:count]
            data_pos = data_pos[:count]
            data_pos_f = data_pos_f[:count]
            data_lin_vel = data_lin_vel[:count]
            data_ang_vel = data_ang_vel[:count]
            data_lin_vel_f = data_lin_vel_f[:count]
            data_ang_vel_f = data_ang_vel_f[:count]
            data_lin_acc = data_lin_acc[:count]
            data_ang_acc = data_ang_acc[:count]
            data_lin_acc_des = data_lin_acc_des[:count]
            data_ang_vel_des = data_ang_vel_des[:count]

            # Save all data to pickle file
            filename='Jetbot_Tracking.pkl'
            # Create dictionary with all collected data and control parameters
            print(f"\nSaving data to {filename}...")
            data_dict = {
            'time': data_time, 
            'pos': data_pos, 
            'pos_f': data_pos_f,
            'lin_vel': data_lin_vel,
            'ang_vel': data_ang_vel,
            'lin_vel_f': data_lin_vel_f,
            'ang_vel_f': data_ang_vel_f,
            'lin_acc': data_lin_acc,
            'ang_acc': data_ang_acc,
            'lin_acc_des': data_lin_acc_des,
            'ang_vel_des': data_ang_vel_des
            }

            # Write dictionary to pickle file
            save_to_pickle(data_dict, filename)
            print("Data saved successfully!")

        print("Done!")
    
def plots():
    data = load_from_pickle('Jetbot_Tracking.pkl')
    t = data["time"]
    pose = data["pos"]
    pose_f = data["pos_f"]
    lin_vel = data["lin_vel"]
    ang_vel = data["ang_vel"]
    lin_vel_f = data["lin_vel_f"]
    ang_vel_f = data["ang_vel_f"]
    lin_acc = data["lin_acc"]
    ang_acc = data["ang_acc"]
    lin_acc_des = data["lin_acc_des"]
    ang_vel_des = data["ang_vel_des"]

    num_bots = pose_f.shape[1]

    # Per agent individual plots
    for i in range(num_bots):
        plot.plot_xy_trajectory(pose_f[:, i, :], title=f"Robot {i} XY Trajectory", show_start_end=True)
        plot.plot_pose_raw_vs_filtered(t, pose_raw=pose[:, i, :], pose_filt=pose_f[:, i, :], title=f"Robot {i} Pose: Raw vs Filtered")
        plot.plot_xy_vs_time(t, pose_f[:, i, :], title=f"Robot {i} Position vs Time (Filtered)")
        plot.plot_velocity_raw_vs_filtered(t, lin_vel[:, i], ang_vel[:, i], lin_vel_f[:, i], ang_vel_f[:, i], title=f"Robot {i} Velocities: Raw vs Filtered")
        plot.plot_velocities(t, lin_vel_f[:, i], ang_vel_f[:, i], v_des=None, w_des=ang_vel_des[:, i], title=f"Robot {i} Velocities vs Time (Filtered)")
        plot.plot_accelerations(t, lin_acc[:, i], ang_acc[:, i], a_des=lin_acc_des[:, i], title=f"Robot {i} Accelerations vs Time", window=30, plot_raw=True)
        plot.plot_accel_and_angvel(t, lin_acc[:, i], ang_vel_f[:, i], lin_acc_des[:, i], ang_vel_des[:, i], title=f"Robot {i} UW actual vs desired")
        
        plt.show()

    # Multiagent plots    
    plot.analyze_dt_histogram(t, bins=30, title="dt Histogram")
    plot.plot_all_xy_trajectories(pose_f, title="All Agents XY Trajectories", labels=["Leader", "Follower1", "Follower2", "Follower3"], show_start_end=True)
    plot.plot_all_linear_velocity(t, lin_vel_f, labels=["Leader", "Follower1", "Follower2", "Follower3"])
    plot.plot_all_angular_velocity(t, ang_vel_f, labels=["Leader", "Follower1", "Follower2", "Follower3"])
    plot.plot_all_linear_acceleration(t, lin_acc, labels=["Leader", "Follower1", "Follower2", "Follower3"], window=20)
    
    plt.show()


if __name__ == "__main__":
    # main()
    plots()