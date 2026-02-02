"""
Lab 6 Part 3: Validation of Camera-workspace Calibration - STARTER CODE
Tracks an AprilTag and transforms its position from camera to workspace frame.
EE 471: Vision-Based Robotic Manipulation
(c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
"""
import numpy as np
import cv2
from time import time

from AprilTags import AprilTags



def main():
    """
    Main validation routine - tracks a tag and displays its position
    """
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    print("="*60)
    print("Camera-workspace Calibration Validation")
    print("="*60)
    
    print("\nInitializing camera and detector...")
    # Initialize camera and detector
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

    # TODO: Fix camera settings
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
    
    detector = AprilTags()  # Replace with actual detector
    
    # TODO: Camera Intrinsics
    fx = 1072.4901458628578
    fy = 1073.7979403880388
    ppx = 322.7882541144218
    ppy = 227.4953665183797

    intrinsics = np.array([
                [fx, 0, ppx],
                [0, fy, ppy],
                [0, 0, 1]])
    print(f"Intrinsics: {intrinsics}")
    
    # Load the calibration transformation matrix
    T_cam_to_workspace = np.load('camera_workspace_transform.npy')  # Replace with loaded transformation
    print("\nLoaded camera-to-workspace transformation matrix:")
    print(T_cam_to_workspace)
    
    # Set the validation tag size in millimeters
    # IMPORTANT: Measure your validation tag!
    TAG_SIZE = 96.5  # Update this value
    # TAG_SIZE = 65  # Update this value

    print(f"\nValidation tag size: {TAG_SIZE} mm")
    
    # Display settings
    PRINT_INTERVAL = 10  # Print every N frames to reduce clutter
    
    print("\n" + "="*60)
    print("Starting validation...")
    print("Move the tag around the workspace")
    print("Press 'q' to quit")
    print("="*60 + "\n")
    
    counter = 0
    
    # =====================================================================
    # MAIN TRACKING LOOP
    # =====================================================================
    while True:
        
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
        
        # Hint: Use detector.detect_tags(color_frame)
        tags = detector.detect_tags(color_frame)  # Replace with detected tags
        
        # -----------------------------------------------------------------
        # STEP 3: PROCESS DETECTED TAGS
        # -----------------------------------------------------------------
        
        # # Check if any tags were detected
        # if len(tags) > 0:
        for tag in tags:

            
            # # Use the first detected tag
            # tag = tags[0]
            
            # Hint: Use detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
            # Returns: (rotation_matrix, translation_vector)
            rot_matrix, trans_vector = detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)

            
            # Check if pose estimation was successful
            if rot_matrix is not None and trans_vector is not None:
                
                # Extract position in camera frame (already in mm)
                # Hint: Flatten trans_vector to get a 1D array of shape (3,)
                pos_camera = trans_vector.flatten()  # Replace with position array

                # Create full 4x4 pose transformation from tag to camera
                T_tag_to_cam = np.eye(4)
                T_tag_to_cam[:3, :3] = rot_matrix  # Replace with rotation matrix
                T_tag_to_cam[:3, 3] = trans_vector.reshape((3,))  # Replace with translation vector
                
                # Transform full pose to workspace frame
                # Hint: Multiply T_cam_to_workspace @ T_tag_to_cam
                T_tag_to_workspace = T_cam_to_workspace @ T_tag_to_cam  # Replace with homogeneous coordinates (4x4)
                
                # Extract position and orientation in workspace frame
                pos_workspace = T_tag_to_workspace[:3, 3]
                rot_workspace = T_tag_to_workspace[:3, :3]

                # Could convert rotation to Euler angles for display
                euler_workspace = cv2.RQDecomp3x3(rot_workspace)[0]
                
                # Calculate distance from camera
                distance = np.linalg.norm(pos_workspace)  # Replace with distance

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


                # Print coordinates periodically to terminal
                if counter % PRINT_INTERVAL == 0:
                    print("\n" + "-"*50)
                    print(f"Tag ID: {tag.tag_id}")
                    print(f"Distance from camera: {distance:.1f} mm")
                    print("\nTag Orientation:")
                    # Print Tag Orientation in workspace Frame
                    print(euler_workspace)
                    
                    print("\nCamera Frame (mm):")
                    # Print 3D positions in Camera Frame
                    print(pos_camera)

                    print("\nworkspace Frame (mm):")
                    # Print 3D positions in workspace Frame
                    print(pos_workspace)

                    print("-"*50)
            
        else:
            # No tags detected
            if counter % PRINT_INTERVAL == 0:
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
        
        # Increment counter
        counter += 1
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting validation...")
            break
    
    print("\nValidation complete!")
    
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()