import numpy as np
import cv2
from time import time
from AprilTags import AprilTags

def setup():
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 535)

    if not cap.isOpened():
        print("Failed to open camera with MSMF")
        exit()

    #fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    return cap

def main():
    # Get camera intrinsic parameters from calibration code
    fx = 487.42056093
    fy = 487.42053388
    cx = 317.3216121
    cy = 248.73120265

    # intrinsics = np.array([
    # [fx, 0, cx],
    # [0, fy, cy],
    # [0,  0,  1]
    # ], dtype=np.float64)

    intrinsics = np.array([fx,fy,cx,cy])

    # Initialize AprilTag detector
    at = AprilTags()
    cap = setup()

    # Tag size in millimeters (measure your actual tag size)
    TAG_SIZE = 65.0  # 65mm tag (update this to match your actual tag size)
    
    # Counter for controlling print frequency
    counter = 0
    last_time = time()
    
    #Construct matrix of tag centeres in the robot base frame

    points_robot = np.array([[665,665,0], #id 11
                            [665,0,0],  #id 57
                            [0,665,0],  #id 61
                            [0,0,0]])    #id 65

    #Take camera measurements
    points_camera = np.zeros((3,4))
    #color_frame, _ = cap.get_frames()
    ret, color_frame = cap.read()
    if not ret:
        print("no image")

    tags = at.detect_tags(color_frame)
    
    iterations = 0
    while (len(tags) != 4) and (iterations < 5):
        print(f"Trying again x{iterations}")
        tags = at.detect_tags(color_frame)
        iterations+=1

    
    #Sort tags with a lambda function
    tags.sort(key=lambda t: t.tag_id)

    for tag in tags:
        # Draw tag detection on image
        color_frame = at.draw_tags(color_frame, tag)

        
        # Get pose estimation
        rot_matrix, trans_vector = at.get_tag_pose(
            tag.corners, 
            intrinsics, 
            TAG_SIZE
        )

        points_camera[:, tag.tag_id] = trans_vector.flatten()
        
    num_points = 4
    T = point_registration(points_camera, points_robot)

    extra = np.ones((1, num_points))
    points_transformed = np.vstack([points_camera, extra])
    bot = np.vstack([points_robot, extra])
    for i in range(points_camera.shape[1]):
        points_transformed[:, i] = T@points_transformed[:, i] #4x4 * 4x5 = 4x5
            
    errors = points_robot - points_transformed[0:3, :]
    error_magnitudes = np.linalg.norm(errors, axis=0)
    print("Error magnitudes: ",error_magnitudes)
    print("Mean error: ",np.mean(error_magnitudes))
    print("Max error: ",np.max(error_magnitudes))
    print("Transformation matrix: ", T)
    
    # Display the image
    cv2.imshow('AprilTag Detection', color_frame)
    np.save('camera_robot_transform.npy', T)
    # Break loop with 'q'
    cv2.waitKey(10)
        

    #Clean up
    #rs.stop()
    #cv2.destroyAllWindows()
        
def point_registration(A, B):
    n = np.shape(A)[1] #= len(B)
    centroidA = np.mean(A, axis=1, keepdims=True)
    centroidB = np.mean(B, axis=1, keepdims=True) #axis one sums across rows for each column

    A = A - centroidA #numpy "broadcasts" smaller matrix repeatedly in bigger matrix
    B = B - centroidB

    H = A @ np.transpose(B) #nxn
    U, S, Vt = np.linalg.svd(H) #3x3, 3x3, 3x3
    V = np.transpose(Vt) #3x3
    Ut = np.transpose(U) #3x3
    R = V @ Ut #3x3
    if np.linalg.det(R) < 0: #reflection if needed 
        negater = np.eye(3)
        negater[2,2] = -1
        V = V @ negater
        R = V @ Ut #3x3
    np.set_printoptions(precision=4, suppress=True)
    #print(R@np.transpose(R))
    #print(np.linalg.det(R).round(4))
    d = centroidB - R@centroidA #3x1
    first = np.hstack([R, d])
    second = np.hstack([np.zeros((1,3)), np.ones((1,1))])
    T = np.vstack([first, second])
    #print(T)
    return T #4 x 4
    


if __name__ == "__main__":
    main()