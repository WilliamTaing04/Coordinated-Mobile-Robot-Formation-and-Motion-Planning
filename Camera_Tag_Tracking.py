import numpy as np
import cv2
import time
import pickle
import AprilTags
import matplotlib.pyplot as plt
from pathlib import Path

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
    

def record():
    """
    Main tracking and recording routine
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
    
    # Define control timestep
    dt = 0.01667  # Control loop period in seconds

    # Pre-allocate arrays for data collection (over-allocate for safety)
    max_samples = 7200     # 60Hz => 60(samples/s)*120s = 7200samples
    data_time = np.zeros(max_samples)                 # Time (s)
    data_ee_pos = np.zeros((max_samples, 3))          # End-effector pose [x,y,z] (mm)
    count = 0  # Sample counter

    begining_time = time.time()  # Record start time for data colleciton

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
        
        if ret is None:
            continue
        
        # -----------------------------------------------------------------
        # STEP 2: DETECT APRILTAGS
        # -----------------------------------------------------------------
        
        # Use detector.detect_tags(color_frame)
        tags = detector.detect_tags(color_frame)  # Replace with detected tags
        
        # -----------------------------------------------------------------
        # STEP 3: PROCESS DETECTED TAGS
        # -----------------------------------------------------------------
        
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

                # Collect Data-----------------------------------------------------
                if count < max_samples:
                    data_time[count] = time.perf_counter() - begining_time  # Time stamps (s)
                    data_ee_pos[count, :] = pos_workspace                   # End-effector pose [x,y,z] (mm)
                    count += 1

                # Draw detection on image
                detector.draw_tags(color_frame, tag)

                corners = np.asarray(tag.corners, dtype=np.float32)  # (4,2)

                # --- Robust "bottom-left-ish" corner under perspective ---
                # 1) find the max y (lowest point)
                max_y = float(np.max(corners[:, 1]))
                # 2) allow a tolerance so rotated tags still get both bottom corners
                tol = 6.0  # pixels; increase if needed
                bottom = corners[corners[:, 1] >= max_y - tol]
                # 3) among bottom candidates, pick the left-most (min x)
                bl = bottom[np.argmin(bottom[:, 0])]
                # Anchor position (start under that corner)
                x = int(bl[0])
                y = int(bl[1] + 5)  # padding below tag
                # Keep on-screen (we'll draw 1 line for ID + 3 lines for xyz)
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
            print("\nNo tag detected - move tag into view")
            
            cv2.putText(color_frame, "No tag detected", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
        
        # -----------------------------------------------------------------
        # STEP 4: DISPLAY AND USER INTERACTION
        # -----------------------------------------------------------------
        
        # Show instruction
        cv2.putText(color_frame, "Press 'q' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Calibration Validation', color_frame)
        
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting validation...")
            break

        # -----------------------------------------------------------------
        # MAINTAIN FIXED TIMESTEP
        # -----------------------------------------------------------------
        
        # Enforce consistent loop timing
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)
    

    # =====================================================================
    # CLEANUP
    # =====================================================================
    print("\nStopping recording and cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.perf_counter() - begining_time
    print(f"\nTotal execution time: {total_time:.2f} s")
    print("Done!")

    # Trim unused portions of pre-allocated arrays
    data_time = data_time[:count]
    data_ee_pos = data_ee_pos[:count, :]

    # Save all data to pickle file
    filename='Camera_Tag_Tracking.pkl'
    # Create dictionary with all collected data and control parameters
    print(f"\nSaving data to {filename}...")
    data_dict = {
    'time': data_time, # Timestamp (s)
    'pos_current': data_ee_pos, # Current EE position [x,y,z] (mm)
    }

    # Write dictionary to pickle file
    save_to_pickle(data_dict, filename)
    print("Data saved successfully!")


def plot_3D_trajectory_list(ee_pose_list, title):
        # Convert lists to arrays
        ee_pose_list = np.array(ee_pose_list)

        # Slice x, y, z, alpha
        x = ee_pose_list[:,0]
        y = ee_pose_list[:,1]
        z = ee_pose_list[:,2]
        # alpha = ee_pose_list[:,4]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z)
        ax.set_title(title)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.show()
    

def plot_xyz_posvelacc_list(t_list, ee_pose_list, title):
        # Convert lists to arrays
        t_list = np.array(t_list)
        ee_pose_list = np.array(ee_pose_list)

        # Slice x, y, z
        x = ee_pose_list[:,0]
        y = ee_pose_list[:,1]
        z = ee_pose_list[:,2]

        # Calculate Velocity and Acceleration
        vx = np.gradient(x)
        vy = np.gradient(y)
        vz = np.gradient(z)
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        az = np.gradient(vz)
        
        # End-effector pose vs time 
        fig, axs = plt.subplots(3,1)    # create subplots
        # Plot position
        axs[0].plot(t_list, x, label='x position')
        axs[0].plot(t_list, y, label='y position')
        axs[0].plot(t_list, z, label='z position')
        axs[0].set_title("XYZ position vs time")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (mm)")
        axs[0].legend(loc='upper right')
        # Plot velocity
        axs[1].plot(t_list, vx, label='x velocity')
        axs[1].plot(t_list, vy, label='y velocity')
        axs[1].plot(t_list, vz, label='z velocity')
        axs[1].set_title("XYZ velocity vs time")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Velocity (mm/s)")
        axs[1].legend(loc='upper right')
        # Plot acceleration
        axs[2].plot(t_list, ax, label='x acceleration')
        axs[2].plot(t_list, ay, label='y acceleration')
        axs[2].plot(t_list, az, label='z acceleration')
        axs[2].set_title("XYZ accerlcation vs time")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Acceleration (mm/s\u00B2)")
        axs[2].legend(loc='upper right')
        
        # Open plot
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()    

def ploting():
    # Load data from pickle files
    data_camtrack = load_from_pickle('Camera_Tag_Tracking.pkl')
    # X step data
    time_pkl_ = data_camtrack["time"]
    ee_pose_pkl = data_camtrack["pos_current"]

    plot_3D_trajectory_list(ee_pose_pkl, "3D Trajectory")
    plot_xyz_posvelacc_list(time_pkl_, ee_pose_pkl, "Position, Velocity, Acceleration")


if __name__ == "__main__":
    record()
    ploting()