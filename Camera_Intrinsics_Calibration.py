
import cv2
import numpy as np

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

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    return cap

# ===== Checkerboard parameters =====
CHECKERBOARD = (7, 4)      # inner corners (cols, rows)
SQUARE_SIZE = 0.0035      # meters (or any consistent unit)
# ===== Calibration flags =====
flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE
objpoints = []  # 3D points
imgpoints = []  # 2D points
cameraIndex = 0

def main():
    images = []
    cap = setup()
    # Get images
    count = 0
    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        status_text = f"Measurements: {count}/{20}"
        cv2.putText(frame, status_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Ready! Press 'q' to capture", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("MSMF Camera", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            images.append(frame)
            count += 1
            cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    print("20 Images Captured")
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            print(f"Skipping")
            continue

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (3, 3),
            (-1, -1),
            criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners)

    N_OK = len(objpoints)
    print(f"Using {N_OK} valid images")

    rvecs = [np.zeros((1, 1, 3)) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3)) for _ in range(N_OK)]

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )
    
    np.set_printoptions(precision=5, suppress=1)    # set print precision and suppression
    print("K:\n", K)
    print("Dist:\n", dist)

    print("\nfx =", K[0,0])
    print("fy =", K[1,1])
    print("ppx =", K[0,2])
    print("ppy =", K[1,2])

    # filename = 'camera_intrinsics.npy'
    # np.save(filename, K)

    # print(K[0,0], K[1,1], K[0,2], K[1,2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
