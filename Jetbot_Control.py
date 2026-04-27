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
import controller
import agent

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
    # Testing
    at_count = 0
    dt_break = 0

    # Settings
    collect_data = True
    control_freq = 30   # Hz
    TAG_SIZE = 96       # mm
    np.set_printoptions(precision=4, suppress=True)


    # Setup camera
    cap = Jetbot_Setup.camera_setup(1280, 720, 0, 30)
    frame_count = 0

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
Leader:
ssh jetbot@10.40.109.62
follower1:
ssh jetbot@10.40.101.192
follower2:
ssh jetbot@10.40.122.94
follower3:
ssh jetbot@10.40.122.89

cd ~/jetbot
python3 -m jetbot.control_reciever
    '''

    # Controllers
    pidvL = Motion_Control.PID(0.5,0.5,0) # PID for v
    pidwL = Motion_Control.PID(1.0,2,0) # PID for w
    pidv1 = Motion_Control.PID(0,0,0) # PID for v
    pidw1 = Motion_Control.PID(0,0,0) # PID for w
    pidv2 = Motion_Control.PID(0,0,0) # PID for v
    pidw2 = Motion_Control.PID(0,0,0) # PID for w
    #pidv3 = Motion_Control.PID(0,0,0) # PID for v
    #pidw3 = Motion_Control.PID(0,0,0) # PID for w
    pidvobs = Motion_Control.PID(0.75,0.5,0) # PID for v
    pidwobs = Motion_Control.PID(1.75,2,0) # PID for w

    # max vel[mm/s], max angvel[rad/s], linmax acc[mm/s^2], send freq, pids
    controllerL = Motion_Control.control(500, 8, 800, control_freq, pidvL, pidwL, alpha=0.95)
    controller1 = Motion_Control.control(500, 8, 800, control_freq, pidv1, pidw1, alpha=0.95)
    controller2 = Motion_Control.control(500, 8, 800, control_freq, pidv2, pidw2, alpha=0.95)
    #controller3 = Motion_Control.control(500, 8, 800, control_freq, pidv3, pidw3, alpha=0.95)
    controllerobs= Motion_Control.control(500, 8, 800, control_freq, pidvobs, pidwobs, alpha=0.95)

    # Jetbots
    leader = Jetbot_Setup.Jetbot(3,"10.40.122.89",controllerL, None, None, role=1,tau_pose=0.01,tau_vel=0.01)   # TagID, 0-follower
    follower1 = Jetbot_Setup.Jetbot(1,"10.40.101.192",controller1, leader, leader, role=0,tau_pose=0.0075,tau_vel=0.0075)   # TagID, 0-follower
    follower1_pred = [0,0] # X, Y edge predecessors. CHANGE HERE AND ABOVE ^ ^ ^ 
    follower2 = Jetbot_Setup.Jetbot(2,"10.40.122.94",controller2, leader, leader, role=0,tau_pose=0.0075,tau_vel=0.0075)   # TagID, 0-follower
    follower2_pred = [0,0] # X, Y edge predecessors. CHANGE HERE AND ABOVE ^ ^ ^ 
    obstacle1 = Jetbot_Setup.Jetbot(4,"10.40.109.62",controllerobs, None, None, role=2,tau_pose=0.0075,tau_vel=0.0075, radius=0.1)
    obstacle2 = Jetbot_Setup.Jetbot(5,"10.40.109.62",controllerobs, None, None, role=2,tau_pose=0.0075,tau_vel=0.0075, radius=0.1)
    obstacle3 = Jetbot_Setup.Jetbot(6,"10.40.109.62",controllerobs, None, None, role=2,tau_pose=0.0075,tau_vel=0.0075, radius=0.1)

    # 11 is 10 61 is 65 65 is 61. ID x_labeled is ID actual
    # follower3 = Jetbot_Setup.Jetbot(9994,"10.40.122.89",controller4,role=0,tau_pose=0.1,tau_vel=0.1)   # TagID, 0-follower

    # Controller params: x_id, y_id, ds_x, ds_y, dsafe_y, gd TODO: state may have to be measured at init
    agentL = None
    agent1 = agent.Agent([0,0,0,0], 1, follower1_pred[0], follower1_pred[1], 3, [-4, -0.5, -0.5], controller.SafeObstacleAvoidanceController(np.array([follower1_pred[0], follower1_pred[1], 0.3, -0.3, -0.05,-4])))
    agent2 = agent.Agent([0,0,0,0], 2, follower2_pred[0], follower2_pred[1], 3, [-4, -0.5, -0.5], controller.SafeObstacleAvoidanceController(np.array([follower2_pred[0], follower2_pred[1], 0.3, 0.3, 0.05,-4])))
    agentobst1 = None
    agentobst2 = None
    agentobst3 = None
    #agent params: state, id, xid, yid, cluster size, estimator gains (gd, gv, p), controller,

    # Jetbot/Agent Arrays
    jetbot_array = [leader, follower1, follower2, obstacle1, obstacle2, obstacle3]
    agent_array = [agentL, agent1, agent2, agentobst1, agentobst2, agentobst3]

    # Desired Leader Movement [m/s] [rad/s] [s]
    # for disturbance test move leaders velocity in oscilatory to show it doesnt propagate through the agents(string stability)
    # leader_time = np.arange(0, 8, 2/20) # start, stop, timestep(N loops at 20Hz)
    # leader_movement = np.zeros((len(leader_time), 3))
    # leader_movement[:,0] = np.sin(leader_time)
    # leader_movement[:,1] = 0.0
    # leader_movement[:,2] = leader_time

    leader_movement = [[100.0, 0.0, 100], #125
                       [0.0, 0.0, 100]]
    
    obstacle1_movement = [[0.0, 0.0, 2]]
    
    leader_move = 0
    obstacle1_move = 0

    if collect_data:
        # Pre-allocate arrays for data collection (over-allocate for safety)
        num_bots = len(jetbot_array)
        max_samples = 7200                              
        data_time = np.zeros(max_samples)                                       # Time [s]
        data_pos   = np.full((max_samples, num_bots, 3), np.nan)                # Jetbot pose [x,y,theta] [mm][rad]
        data_pos_f = np.full((max_samples, num_bots, 3), np.nan)                # Jetbot pose [x,y,theta] [mm][rad] (filtered)
        data_lin_vel = np.zeros((max_samples, num_bots))                        # Jetbot lin velocity [mm/s]
        data_ang_vel = np.zeros((max_samples, num_bots))                        # Jetbot ang velocity [rad/s]
        data_lin_vel_f = np.zeros((max_samples, num_bots))                      # Jetbot lin velocity [mm/s] (filtered)
        data_ang_vel_f = np.zeros((max_samples, num_bots))                      # Jetbot ang velocity [rad/s] (filtered)
        data_lin_acc = np.zeros((max_samples, num_bots))                        # Jetbot lin acceleration [mm/s^2]
        data_ang_acc = np.zeros((max_samples, num_bots))                        # Jetbot ang acceleration [rad/s^2]
        data_lin_acc_des = np.zeros((max_samples, num_bots))                    # Agent lin acceleration [mm/s^2] (desired)
        data_ang_vel_des = np.zeros((max_samples, num_bots))                    # Agent ang velocity [rad/s] (desired)
        data_long_sb = np.zeros((max_samples, num_bots))                        # Agent longitudinal safety barrier [m]   (h1)
        data_lat_sb = np.zeros((max_samples, num_bots))                         # Agent latitudinal safety barrier [m]    (h2)
        data_form_dist_along = np.zeros((max_samples, num_bots))                # Agent formation distance along motion [m]
        data_form_dist_perp = np.zeros((max_samples, num_bots))                 # Agent formation distance perpendicular to motion [m]
        data_long_des = np.zeros((max_samples, num_bots))                       # Agent desired formation distance along motion(x axis) [m]   (ds_x = dx_star)
        data_lat_des = np.zeros((max_samples, num_bots))                        # Agent desired formation distance perpendicular to motion(y axis) [m]    (ds_y = dy_star)
        data_long_safe_limit = np.zeros((max_samples, num_bots))                # Agent safety limit along motion(x axis) [m]   (dsafe_x)
        data_lat_safe_limit = np.zeros((max_samples, num_bots))                 # Agent safety limit perpendicular to motion(y axis) [m]    (dsafe_y)
        data_leader_pos_est = np.zeros((max_samples, num_bots))                 # Agent estimate of both leaders' pose(x,y) [m]
        data_leader_vel_est = np.zeros((max_samples, num_bots))                 # Agent estimate of both leaders' velocity [m/s]   
        
        count = 0  # Sample counter

    initial_time = time.perf_counter()
    leader_move_start = time.perf_counter()
    obstacle1_move_start = time.perf_counter()

# =====================================================================
# MAIN TRACKING LOOP
# =====================================================================
    try:
        while True:
            # Record start time of the loop
            start_time = time.perf_counter()
            frame_count += 1
            for jetbot in jetbot_array:
                jetbot.visible = 0   # reset visibility

            # CAPTURE FRAME
            ret, color_frame = cap.read()
            if not ret:
                continue

            # STEP 2: DETECT APRILTAGS            
            tags = detector.detect_tags(color_frame)  # Replace with detected tags
            
            # STEP 3: PROCESS DETECTED TAGS
            # If no tags are detected 
            if len(tags)==0:
                # No tags detected      
                if (frame_count % 4) == 0:      
                    cv2.putText(color_frame, "No tag detected", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2)

            else: 
                tags.sort(reverse=True) # Sort tags to match jetbot array TODO: may not be nessecary anymore
                for tag in tags:
                    # Get tag pose                    
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

                        # Update jetbot pose, agent rk4 and alg
                        for i, jetbot in enumerate(jetbot_array):
                            if (tag_id == jetbot.id):   # filter out bad tags
                                t_meas = time.perf_counter()
                                jetbot.update_meas(pose, t_meas)    # update jetbot
                                jetbot.visible = 1                  # jetbot seen
                                if(Jetbot_Setup.check_init(jetbot_array)) and (jetbot.role==0):   # check if all poses are initialized
                                    d_X, v_X, theta_X = jetbot.get_dist_theta(jetbot.X_lead) # [mm, mm/s, radians]
                                    d_Y, v_Y, theta_Y = jetbot.get_dist_theta(jetbot.Y_lead) # [mm, mm/s, radians]
                                    X_upd = np.array([d_X/1000, v_X/1000, theta_X]) # mm to m [m, m/s, radians]
                                    Y_upd = np.array([d_Y/1000, v_Y/1000, theta_Y]) # mm to m [m, m/s, radians]
                                    agent_array[i].update_edges(X_upd,Y_upd)
                                    agent_array[i].init_estimates() # only runs the first time

                                    if isinstance(agent_array[i].controller, controller.SafeObstacleAvoidanceController):   # checks controller type
                                        #UPDATE OBSTACLES:
                                        agent_array[i].controller.obstacle_data = jetbot.get_obst_meas(jetbot_array)    # pass obstacle measurements to agent

                    # Reduce display output
                    if (frame_count % 5) == 0:
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


            # STEP 4: CONTROLLER AND COMMUNICATION
            # Jetbot Motion Control and Algorithm
            for i, jetbot in enumerate(jetbot_array):
                # Follower Control
                if jetbot.visible and jetbot.role==0: # For followers
                    agent_array[i].RK4_step() # RK4 step good
                    U_GOAL, W_GOAL = agent_array[i].get_controls() # Get goal UW from alg
                    
                    # Record agent data
                    data_lin_acc_des[count-1, i] = U_GOAL * 1000  # m/s^2 -> mm/s^2
                    data_ang_vel_des[count-1, i] = W_GOAL
                    data_long_sb[count-1, i] = agent_array[i].controller.h1
                    data_lat_sb[count-1, i] = agent_array[i].controller.h2
                    data_long_des[count-1, i] = agent_array[i].controller.ds_x
                    data_lat_des[count-1, i] = agent_array[i].controller.ds_y
                    data_long_safe_limit[count-1, i] = agent_array[i].controller.dsafe_x
                    data_lat_safe_limit[count-1, i] = agent_array[i].controller.dsafe_y
                    data_leader_pos_est[count-1, i] = math.hypot(agent_array[i].agent_metadata[0][0][0], agent_array[i].agent_metadata[0][0][2]) - agent_array[i].agent_metadata[1][0][0]
                    data_leader_vel_est[count-1, i] = math.hypot(agent_array[i].agent_metadata[0][0][1], agent_array[i].agent_metadata[0][0][3]) - (leader.lin_vel_f - jetbot.lin_vel_f)
                    data_form_dist_along[count-1, i] = agent_array[i].agent_metadata[1][0][0] * np.cos(agent_array[i].agent_metadata[1][0][1])
                    data_form_dist_perp[count-1, i] = agent_array[i].agent_metadata[1][0][0] * np.sin(agent_array[i].agent_metadata[1][0][1])

                    t_now = time.perf_counter()
                    # UW controller
                    v_cmd , w_cmd = jetbot.controller.controller_uw([jetbot.lin_vel_f, jetbot.ang_vel_f],[U_GOAL*1000, W_GOAL])
                    # Convert Desired VW to LR motor speed
                    left, right = jetbot.controller.motor_controller(v_cmd, w_cmd)
                
                elif jetbot.visible and jetbot.role==1: # For leaders
                    if leader_move < len(leader_movement):
                        leader_v, leader_w, move_duration = leader_movement[leader_move]
                        if time.perf_counter() - leader_move_start < move_duration:
                            # data_lin_acc_des[count-1, i] = None # TODO: maybe record leader desired lin vel
                            # data_ang_vel_des[count-1, i] = leader_w
                            v_cmd, w_cmd = jetbot.controller.controller_vw([jetbot.lin_vel_f, jetbot.ang_vel_f], [leader_v, leader_w])
                            # Convert Desired VW to LR motor speed
                            left, right = jetbot.controller.motor_controller(v_cmd, w_cmd)
                        else:
                            leader_move += 1
                            leader_move_start = time.perf_counter()
                    else:
                        left = right = 0.0

                # obstacle movement:
                elif jetbot.visible and jetbot.role==2: # For obstacles
                    if jetbot.id == 999:   # TODO: change with obstacle id
                        if obstacle1_move < len(obstacle1_movement):
                            obstacle_v, obstacle_w, move_duration = obstacle1_movement[obstacle1_move]
                            if time.perf_counter() - obstacle1_move_start < move_duration:
                                data_lin_acc_des[count-1, i] = None
                                data_ang_vel_des[count-1, i] = None
                                v_cmd, w_cmd = jetbot.controller.controller_vw([jetbot.lin_vel_f, jetbot.ang_vel_f], [obstacle_v, obstacle_w])
                                # Convert Desired VW to LR motor speed
                                left, right = jetbot.controller.motor_controller(v_cmd, w_cmd)
                            else:
                                obstacle1_move += 1
                                obstacle1_move_start = time.perf_counter()
                        else:
                            left = right = 0.0

                else:   # If jetbot is not visible then stop movement
                    left = right = 0.0
                
                # Send command to jetbot
                UDP.Send(jetbot.IP, left, right)

            at_count += 1 # TESTING at frame count

            # Reduce display
            if (frame_count % 5) == 0:
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

            # MAINTAIN FIXED TIMESTEP
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
        # CLEANUP
        print("\n[STOP] Sending stop command and exiting...")
        for jetbot in jetbot_array:
            UDP.Close(jetbot.IP)    # disconnect from UDP
        UDP.Shutdown()  # shutdown socket
        cap.release()   # exit camera
        cv2.destroyAllWindows() # close windows
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
            data_long_sb = data_long_sb[:count]
            data_lat_sb = data_lat_sb[:count]
            data_long_des = data_long_des[:count]
            data_lat_des = data_lat_des[:count]
            data_long_safe_limit = data_long_safe_limit[:count]
            data_lat_safe_limit = data_lat_safe_limit[:count]
            data_leader_pos_est = data_leader_pos_est[:count]
            data_leader_vel_est = data_leader_vel_est[:count]
            data_form_dist_along = data_form_dist_along[:count]
            data_form_dist_perp = data_form_dist_perp[:count]


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
            'ang_vel_des': data_ang_vel_des,
            'long_sb': data_long_sb, 
            'lat_sb': data_lat_sb,
            'long_des': data_long_des,
            'lat_des': data_lat_des,
            'long_safe_limit': data_long_safe_limit,
            'lat_safe_limit': data_lat_safe_limit,
            'leader_pos_est': data_leader_pos_est,
            'leader_vel_est': data_leader_vel_est,
            'form_dist_along': data_form_dist_along,
            'form_dist_perp': data_form_dist_perp
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
    data_long_sb = data["long_sb"]
    data_lat_sb = data["lat_sb"]
    data_long_des = data["long_des"]
    data_lat_des = data["lat_des"]
    data_long_safe_limit = data["long_safe_limit"]
    data_lat_safe_limit = data["lat_safe_limit"]
    data_leader_pos_est = data["leader_pos_est"]
    data_leader_vel_est = data["leader_vel_est"]
    data_form_dist_along = data['form_dist_along']
    data_form_dist_perp = data['form_dist_perp']
    num_bots = pose_f.shape[1]
    


    # Per agent individual plots
    # for i in range(num_bots):
    #     plot.plot_xy_trajectory(pose_f[:, i, :], title=f"Robot {i} XY Trajectory", show_start_end=True)
    #     # plot.plot_pose_raw_vs_filtered(t, pose_raw=pose[:, i, :], pose_filt=pose_f[:, i, :], title=f"Robot {i} Pose: Raw vs Filtered")
    #     plot.plot_xy_vs_time(t, pose_f[:, i, :], title=f"Robot {i} Position vs Time (Filtered)")
    #     # plot.plot_velocity_raw_vs_filtered(t, lin_vel[:, i], ang_vel[:, i], lin_vel_f[:, i], ang_vel_f[:, i], title=f"Robot {i} Velocities: Raw vs Filtered")
    #     plot.plot_velocities(t, lin_vel_f[:, i], ang_vel_f[:, i], v_des=None, w_des=ang_vel_des[:, i], title=f"Robot {i} Velocities vs Time (Filtered)")
    #     plot.plot_accelerations(t, lin_acc[:, i], ang_acc[:, i], a_des=lin_acc_des[:, i], title=f"Robot {i} Accelerations vs Time", window=30, plot_raw=True)
    #     plot.plot_accel_and_angvel(t, lin_acc[:, i], ang_vel_f[:, i], lin_acc_des[:, i], ang_vel_des[:, i], title=f"Robot {i} UW actual vs desired")
    # plt.show()

    # Multiagent plots    
    # plot.analyze_dt_histogram(t, bins=30, title="dt Histogram")
    plot.plot_all_xy_trajectories(pose_f, title="All Agents XY Trajectories", labels=["Leader", "Follower 1", "Follower 2", "Obstable 1", "Obstable 2", "Obstable 3"], show_start_end=True)
    plot.plot_all_linear_velocity(t, lin_vel_f, labels=["Leader", "Follower 1", "Follower 2"])
    plot.plot_all_angular_velocity(t, ang_vel_f, labels=["Leader", "Follower 1", "Follower 2"])
    plot.plot_all_linear_acceleration(t, lin_acc, labels=["Leader", "Follower 1", "Follower 2"], window=20)

    plot.basic_plot(t, data_leader_pos_est,"Time (s)", "Estimation vs Absolute (m)","Leader Position Estimation vs Absolute Error")
    plot.basic_plot(t, data_leader_vel_est,"Time (s)", "Estimation vs Absolute (m/s)","Leader Velocity Estimation vs Absolute Error")

    plt.show()


if __name__ == "__main__":
    # main()
    plots()