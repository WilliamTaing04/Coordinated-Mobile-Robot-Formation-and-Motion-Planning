import time
import numpy as np
import cv2
import math
import AprilTags 
import Motion_Control
import Jetbot_Setup

def testing():
    cap = Jetbot_Setup.camera_setup(1280, 720)
    detector = AprilTags.AprilTags()

    intrinsics = np.load('camera_intrinsics.npy')  # Replace with loaded transformation
    T_cam_to_workspace = np.load('camera_workspace_transform.npy')  # Replace with loaded transformation

    control_freq = 30

    # Controllers
    pidvL = Motion_Control.PID(0.75,0.5,0) # PID for v
    pidwL = Motion_Control.PID(1.75,2,0) # PID for w
    pidv1 = Motion_Control.PID(0,0,0) # PID for v
    pidw1 = Motion_Control.PID(0,0,0) # PID for w
    pidv2 = Motion_Control.PID(0,0,0) # PID for v
    pidw2 = Motion_Control.PID(0,0,0) # PID for w
    # pidv3 = Motion_Control.PID(0,0,0) # PID for v
    # pidw3 = Motion_Control.PID(0,0,0) # PID for w

    # max vel[mm/s], max angvel[rad/s], linmax acc[mm/s^2], send freq, pids
    controllerL = Motion_Control.control(500, 8, 800, control_freq, pidvL, pidwL, alpha=0.05)
    controller1 = Motion_Control.control(500, 8, 800, control_freq, pidv1, pidw1, alpha=0.05)
    controller2 = Motion_Control.control(500, 8, 800, control_freq, pidv2, pidw2, alpha=0.05)
    # controller3 = Motion_Control.control(500, 8, 800, control_freq, pidv3, pidw3, alpha=0.05)

    # Jetbots
    leader = Jetbot_Setup.Jetbot(26,"10.40.109.62",controllerL, None, None, role=1,tau_pose=0.01,tau_vel=0.01)   # TagID, 0-follower
    follower1 = Jetbot_Setup.Jetbot(11,"10.40.101.192",controller1, leader, leader, role=0,tau_pose=0.0075,tau_vel=0.0075)   # TagID, 0-follower
    follower2 = Jetbot_Setup.Jetbot(9,"10.40.122.94",controller2, leader, follower1, role=0,tau_pose=0.0075,tau_vel=0.0075)   # TagID, 0-follower
    # follower3 = Jetbot_Setup.Jetbot(9994,"10.40.122.89",controller4,role=0,tau_pose=0.1,tau_vel=0.1)   # TagID, 0-follower    

    # Jetbot/Agent Arrays
    jetbot_array = [leader, follower1, follower2]

    while True:
        pose = update_pose(jetbot_array, cap, detector, intrinsics, T_cam_to_workspace)
        print(pose)
        time.sleep(0.1)




def update_pose(jetbot_array, cap, detector, intrinsics, T_cam_to_workspace):
    output = np.zeros((len(jetbot_array),4))
    TAG_SIZE = 65       # mm

    for jetbot in jetbot_array:
        jetbot.visible = 0   # reset visibility

    # CAPTURE FRAME
    ret, color_frame = cap.read()
    if not ret:
        pass

    # STEP 2: DETECT APRILTAGS            
    tags = detector.detect_tags(color_frame)  # Replace with detected tags
    print(len(tags))
    # STEP 3: PROCESS DETECTED TAGS
    # If no tags are detected 
    if len(tags)==0:
        # No tags detected      
        cv2.putText(color_frame, "No tag detected", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
    
        
    else: 
        tags.sort(reverse=True) # Sort tags to match jetbot array
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
                # Update jetbot pose
                for i, jetbot in enumerate(jetbot_array):
                    
                    if (tag_id == jetbot.id):
                        t_meas = time.perf_counter()
                        jetbot.update_meas(pose, t_meas)
                        jetbot.visible = 1
                        output[i,:] = [jetbot.pose_f[0],jetbot.pose_f[1],jetbot.lin_vel_f,jetbot.pose_f[2]]

    return(output)

    #cv2.putText(color_frame, "Press 'q' to quit", 
                #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               # 0.7, (0, 255, 0), 2)
    
    # Display frame
    #cv2.imshow('Camera', color_frame)




def command_jetbots(UDP, jetbot_array, UW_goals):
    for i, jetbot in enumerate(jetbot_array):
            # Follwer Control
            if jetbot.visible and jetbot.role==0:
                
                U_GOAL, W_GOAL = UW_goals[i]     # Get goal UW from alg
                # Record desired UW
                # data_lin_acc_des[count-1, i] = U_GOAL * 1000  # m/s^2 -> mm/s^2
                # data_ang_vel_des[count-1, i] = W_GOAL
                t_now = time.perf_counter()
                # UW controller
                v_cmd , w_cmd = jetbot.controller.controller_uw([jetbot.lin_vel, jetbot.ang_vel],[U_GOAL*1000, W_GOAL])
                # Convert Desired VW to LR motor speed
                left, right = jetbot.controller.motor_controller(v_cmd, w_cmd)
            
            # elif jetbot.visible and jetbot.role==1:
            #     if move < len(leader_movement):
            #         leader_v, leader_w, move_duration = leader_movement[move]
            #         if time.perf_counter() - move_start < move_duration:
            #             data_lin_acc_des[count-1, i] = None
            #             data_ang_vel_des[count-1, i] = leader_w
            #             # VW controller
            #             v_cmd, w_cmd = jetbot.controller.controller_vw([jetbot.lin_vel, jetbot.ang_vel], [leader_v, leader_w])
            #             # Convert Desired VW to LR motor speed
            #             left, right = jetbot.controller.motor_controller(v_cmd, w_cmd)
            #         else:
            #             move += 1
            #             move_start = time.perf_counter()
            #     else:
            #         left = right = 0.0

            else:   # If jetbot is not visible then stop movement
                left = right = 0.0
            UDP.Send(jetbot.IP, left, right)


if __name__ == "__main__":
    testing()