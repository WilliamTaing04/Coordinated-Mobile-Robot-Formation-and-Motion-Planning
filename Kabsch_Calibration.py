"""
Implements camera-workspace calibration using AprilTags and the Kabsch algorithm.
EE 471: Vision-Based Robotic Manipulation
(c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
"""
import numpy as np
import cv2
from time import time, sleep
from AprilTags import AprilTags


def point_registration(points_A, points_B):
    """
    Implement Kabsch algorithm to find optimal rigid transformation between point sets.
    
    This is the same algorithm from Pre-Lab 6, computing the transformation that
    maps points from coordinate system A to coordinate system B.
    
    Args:
        points_A: 3xN array of points in frame A (camera frame)
        points_B: 3xN array of corresponding points in frame B (workspace frame)
        
    Returns:
        4x4 homogeneous transformation matrix from A to B
    """
    assert points_A.shape == points_B.shape, "Point sets must have same dimensions"
    assert points_A.shape[0] == 3, "Points must be 3D"
    
    # Compute centroids
    centroid_A = np.mean(points_A, axis=1, keepdims=True)
    centroid_B = np.mean(points_B, axis=1, keepdims=True)
    
    # Center the point sets
    A_centered = points_A - centroid_A
    B_centered = points_B - centroid_B
    
    # Compute cross-covariance matrix
    H = A_centered @ B_centered.T
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        print("  Warning: Reflection detected, correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_B - R @ centroid_A
    
    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = t
    
    return T

def main():
    """
    Main calibration routine.
    """
    # =====================================================================
    # INITIALIZATION
    # =====================================================================
    print("="*60)
    print("Camera Calibration")
    print("="*60)
    
    # Initialize camera and detector
    print("\nInitializing camera and AprilTag detector...")
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

    detector = AprilTags()
    
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
    
    # IMPORTANT: Measure your actual tags!
    TAG_SIZE = 96.5  # Default tag size
    print(f"Tag size: {TAG_SIZE} mm")
    
    # =====================================================================
    # DEFINE WORKSPACE FRAME REFERENCE POINTS
    # =====================================================================
    print("\nDefining workspace frame reference points...")
    
    # Define measured tag positions in workspace frame
    # IMPORTANT: Replace these with YOUR measured coordinates!
    # Measure each tag center position relative to workspace base origin
    # Store in order: Tag ID 0, 1, 2, ..., 11 (left-to-right, top-to-bottom)
    ztag=np.array([0, 0])      # position of center of tag 0 in workspace frame
    xspace0 = 48 + TAG_SIZE    # x spacing between pairs
    xspace1 = 38 + TAG_SIZE    # x spacing between papers
    yspace = 119.5 + TAG_SIZE  # y spacing between papers
    # x = 0, 144.5 , 279.0 , 423.5 , 558.0, 702.5
    # y = 0, -216.0 , -432,0 

    workspace_points_array = np.array([
        # [X, Y, Z] coordinates in mm for each tag
        # Paper 1 (tags 0-1)
        [ztag[0] + (0*xspace0) + (0*xspace1), ztag[1] - (0*yspace), 0],      # Tag 0
        [ztag[0] + (1*xspace0) + (0*xspace1), ztag[1] - (0*yspace), 0],      # Tag 1
        # Paper 2 (tags 2-3)
        [ztag[0] + (1*xspace0) + (1*xspace1), ztag[1] - (0*yspace), 0],      # Tag 2
        [ztag[0] + (2*xspace0) + (1*xspace1), ztag[1] - (0*yspace), 0],      # Tag 3
        # Paper 3 (tags 4-5)
        [ztag[0] + (2*xspace0) + (2*xspace1), ztag[1] - (0*yspace), 0],      # Tag 4
        [ztag[0] + (3*xspace0) + (2*xspace1), ztag[1] - (0*yspace), 0],      # Tag 5
        # Paper 4 (tags 6-7)
        [ztag[0] + (0*xspace0) + (0*xspace1), ztag[1] - (1*yspace), 0],      # Tag 6
        [ztag[0] + (1*xspace0) + (0*xspace1), ztag[1] - (1*yspace), 0],      # Tag 7
        # Paper 5 (tags 8-9)
        [ztag[0] + (1*xspace0) + (1*xspace1), ztag[1] - (1*yspace), 0],      # Tag 8
        [ztag[0] + (2*xspace0) + (1*xspace1), ztag[1] - (1*yspace), 0],      # Tag 9
        # Paper 6 (tags 10-11)
        [ztag[0] + (2*xspace0) + (2*xspace1), ztag[1] - (1*yspace), 0],      # Tag 10
        [ztag[0] + (3*xspace0) + (2*xspace1), ztag[1] - (1*yspace), 0],      # Tag 11
        # Paper 7 (tags 12-13)
        [ztag[0] + (0*xspace0) + (0*xspace1), ztag[1] - (2*yspace), 0],      # Tag 12
        [ztag[0] + (1*xspace0) + (0*xspace1), ztag[1] - (2*yspace), 0],      # Tag 13
        # Paper 8 (tags 14-15)
        [ztag[0] + (1*xspace0) + (1*xspace1), ztag[1] - (2*yspace), 0],      # Tag 14
        [ztag[0] + (2*xspace0) + (1*xspace1), ztag[1] - (2*yspace), 0],      # Tag 15
        # Paper 9 (tags 16-17)
        [ztag[0] + (2*xspace0) + (2*xspace1), ztag[1] - (2*yspace), 0],      # Tag 16
        [ztag[0] + (3*xspace0) + (2*xspace1), ztag[1] - (2*yspace), 0],      # Tag 17
    ])
    print("testing: ")
    print(workspace_points_array)

    # Convert to 3xN format (transpose)
    points_workspace = workspace_points_array.T  # Shape: (3, 36)
    num_tags = points_workspace.shape[1]
    print(f"Expecting {num_tags} tags (IDs 0-{num_tags-1})")
    
    # Initialize camera points array (3 x num_tags, filled with zeros)
    points_camera = np.zeros((3, num_tags))
    # Create a list of empty lists, one for each tag
    measurements = [[] for _ in range(num_tags)]
    
    # =====================================================================
    # COLLECT MEASUREMENTS
    # =====================================================================
    print("\n" + "="*60)
    print("Starting measurement collection...")
    print("Position camera to see all tags clearly.")
    print("Press 'c' to capture a measurement, 'q' to finish")
    print("="*60)
    
    num_measurements = 0
    target_measurements = 5  # Collect 5 measurements for averaging
    
    while num_measurements < target_measurements:
        
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
        
        # Detect AprilTags in the frame
        # Hint: Use detector.detect_tags(color_frame)
        # YOUR CODE HERE
        tags = detector.detect_tags(color_frame)  # Replace with detected tags
        
        # -----------------------------------------------------------------
        # STEP 3: VISUALIZE DETECTIONS
        # -----------------------------------------------------------------
        
        # Create a copy for display
        display_frame = color_frame.copy()
        
        # Draw all detected tags on display_frame
        for tag in tags:
            display_frame = detector.draw_tags(display_frame, tag)
        
        # Status overlay
        status_text = f"Tags detected: {len(tags)}/{num_tags} | Measurements: {num_measurements}/{target_measurements}"
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(tags) == num_tags:
            cv2.putText(display_frame, "Ready! Press 'c' to capture", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Align camera to see all tags", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Calibration', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # -----------------------------------------------------------------
        # STEP 4: CAPTURE MEASUREMENT ON USER INPUT
        # -----------------------------------------------------------------
        
        if key == ord('c'):  # User pressed 'c' to capture
            
            if len(tags) == num_tags:
                print(f"\nCapturing measurement {num_measurements + 1}/{target_measurements}...")
                
                # Sort tags by ID to maintain correspondence with workspace_points
                tags_sorted = sorted(tags, key=lambda t: t.tag_id)
                
                # Process each tag to get its pose
                temp_measurements = []
                
                for tag in tags_sorted:
                    
                    # Get pose estimation for this tag
                    # Hint: Use detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                    # Returns: (rotation_matrix, translation_vector)
                    # Note: We only need translation_vector (tag position) for calibration
                    # YOUR CODE HERE
                    rot_matrix, trans_vector = detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                    print(trans_vector)
                    
                    # Store the translation vector (position in camera frame)
                    # Hint: Flatten trans_vector and append to temp_measurements
                    # Note: We ignore rot_matrix - only position is needed for Kabsch algorithm
                    # YOUR CODE HERE
                    temp_measurements.append(trans_vector.flatten())


                # If all tags processed successfully, store measurements
                # Loop through temp_measurements and append each to measurements[idx]
                for idx, measurement in enumerate(temp_measurements):
                    measurements[idx].append(measurement)
                num_measurements += 1
                
            else:
                print(f"  Error: Only {len(tags)}/{num_tags} tags visible. Need all tags!")
        
        # Quit on 'q'
        elif key == ord('q'):
            if num_measurements > 0:
                print(f"\nFinishing with {num_measurements} measurements...")
                break
            else:
                print("\nNo measurements collected. Exiting...")
                return
    
    # =====================================================================
    # PROCESS MEASUREMENTS
    # =====================================================================
    print("\n" + "="*60)
    print("Processing measurements...")
    
    # Average all measurements for each tag
    for idx, tag_measurements in enumerate(measurements):
        if len(tag_measurements) > 0:
            avg_position = np.mean(tag_measurements, axis=0)
            points_camera[:, idx] = avg_position
            print(f"  Tag {idx}: {len(tag_measurements)} measurements averaged")
        else:
            print(f"  Warning: No measurements for tag {idx}!")
    print(points_camera)
    print(points_workspace)
    # =====================================================================
    # COMPUTE TRANSFORMATION
    # =====================================================================
    print("\nComputing camera-to-workspace transformation...")
    
    # Call point_registration() to compute transformation
    # YOUR CODE HERE
    T_cam_to_workspace = point_registration(points_camera, points_workspace)  # Replace with actual transformation
    
    print("\nTransformation matrix (camera to workspace):")
    print(T_cam_to_workspace)
    
    # Extract rotation and translation for display
    # YOUR CODE HERE
    R = T_cam_to_workspace[:3, :3] # Replace with actual rotation
    t = T_cam_to_workspace[:3, 3] # Replace with actual translation

    print(f"\nTranslation: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] mm")
    
    # Verify rotation matrix properties
    det_R = np.linalg.det(R)
    orthogonality_check = np.linalg.norm(R @ R.T - np.eye(3))
    print(f"Rotation matrix determinant: {det_R:.6f} (should be 1.0)")
    print(f"Orthogonality error: {orthogonality_check:.6e} (should be ~0)")
    
    # =====================================================================
    # CALCULATE CALIBRATION ERROR
    # =====================================================================
    print("\n" + "="*60)
    print("Calculating calibration accuracy...")
    
    # Transform camera points to workspace frame using T_cam_to_workspace
    # Hint: Convert points_camera to homogeneous coordinates first (add row of ones)
    # Then multiply: T_cam_to_workspace @ points_camera_homogeneous
    # YOUR CODE HERE
    points_camera_hom = np.vstack((points_camera, np.ones((1,num_tags), dtype=float)))  # Replace with homogeneous coordinates
    points_camera_transformed = T_cam_to_workspace @ points_camera_hom  # Replace with transformed points
    
    # Calculate errors between transformed camera points and workspace points
    # Hint: Compute differences, then use np.linalg.norm() on each column
    # YOUR CODE HERE
    errors = points_camera_transformed[:3] - points_workspace  # Replace with error vectors

    error_magnitudes = np.linalg.norm(errors, axis=0)
    
    # Calculate error statistics
    # Hint: Use np.mean(), np.std(), np.max(), np.min()
    # YOUR CODE HERE
    mean_error = np.mean(error_magnitudes)  # Replace with mean
    std_error = np.std(error_magnitudes)  # Replace with std
    max_error = np.max(error_magnitudes)  # Replace with max
    min_error = np.min(error_magnitudes)  # Replace with min
    
    print(f"\nCalibration Error Statistics:")
    print(f"  Mean error:    {mean_error:.3f} mm")
    print(f"  Std deviation: {std_error:.3f} mm")
    print(f"  Min error:     {min_error:.3f} mm")
    print(f"  Max error:     {max_error:.3f} mm")
    
    # Print per-tag errors
    print(f"\nPer-tag errors:")
    for i in range(num_tags):
        print(f"  Tag {i:2d}: {error_magnitudes[i]:6.3f} mm")
    
    # Quality assessment
    if mean_error < 5.0:
        print("\n✓ Calibration quality: EXCELLENT (< 5 mm)")
    elif mean_error < 10.0:
        print("\n✓ Calibration quality: GOOD (< 10 mm)")
    else:
        print("\n⚠ Calibration quality: ACCEPTABLE but consider recalibrating")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    print("\n" + "="*60)
    
    # Save transformation matrix to file
    filename = 'C:/Users/cmcgarit/Desktop/SP_Code/camera_workspace_transform.npy'
    # YOUR CODE HERE
    np.save(filename, T_cam_to_workspace)
    

    print(f"Transformation matrix saved to '{filename}'")
    print("="*60)
    
    print("\nCleaning up...")
    # Stop camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    print("Done!")


if __name__ == "__main__":
    main()