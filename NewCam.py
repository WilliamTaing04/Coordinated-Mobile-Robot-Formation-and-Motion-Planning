import numpy as np
import cv2
import time
import pickle
import AprilTags
import matplotlib.pyplot as plt
from pathlib import Path

def record():
    print("="*60)
    print("Workspace AprilTag Tracking")
    print("="*60)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    # ---- Camera performance settings ----
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera resolution: {width} x {height}")
    print(f"Camera FPS (reported): {fps}")


    detector = AprilTags.AprilTags()

    # ---- Camera intrinsics ----
    fx, fy = 490.00332243, 489.5556459
    ppx, ppy = 315.8040739, 268.93739803
    intrinsics = np.array([[fx, 0, ppx],
                           [0, fy, ppy],
                           [0,  0,   1]])

    T_cam_to_workspace = np.load('camera_workspace_transform.npy')
    TAG_SIZE = 65  # mm

    # ---- Data buffers ----
    max_samples = 7200
    data_time = np.zeros(max_samples)
    data_ee_pos = np.zeros((max_samples, 3))
    count = 0

    start_global = time.perf_counter()

    # ---- FPS tracking ----
    from collections import deque
    frame_times = deque(maxlen=30)
    last_frame_time = time.perf_counter()

    # ---- Text settings (reuse!) ----
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("\nTracking started — press 'q' to quit\n")

    while True:
        now = time.perf_counter()
        ret, color_frame = cap.read()
        if not ret:
            continue

        # ---- FPS update ----
        frame_times.append(now - last_frame_time)
        last_frame_time = now
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if len(frame_times) > 1 else 0.0

        # ---- AprilTag detection ----
        tags = detector.detect_tags(color_frame)

        if tags:
            tag = tags[0]
            rot_matrix, trans_vector = detector.get_tag_pose(
                tag.corners, intrinsics, TAG_SIZE
            )

            if rot_matrix is not None:
                T_tag_to_cam = np.eye(4)
                T_tag_to_cam[:3, :3] = rot_matrix
                T_tag_to_cam[:3, 3] = trans_vector.ravel()

                T_tag_to_workspace = T_cam_to_workspace @ T_tag_to_cam
                pos_workspace = T_tag_to_workspace[:3, 3]

                if count < max_samples:
                    data_time[count] = now - start_global
                    data_ee_pos[count] = pos_workspace
                    count += 1

                detector.draw_tags(color_frame, tag)

                # ---- Corner-anchored text ----
                corners = np.asarray(tag.corners, dtype=np.float32)
                max_y = np.max(corners[:, 1])
                bottom = corners[corners[:, 1] >= max_y - 6.0]
                bl = bottom[np.argmin(bottom[:, 0])]

                x, y = int(bl[0]), int(bl[1] + 5)
                h, w = color_frame.shape[:2]
                x = max(5, min(x, w - 140))
                y = max(15, min(y, h - 60))

                cv2.putText(color_frame, f"Tag ID: {tag.tag_id}",
                            (x, y), font, 0.35, (0, 255, 0), 1)

                for i, lab in enumerate(("x", "y", "z")):
                    cv2.putText(
                        color_frame,
                        f"{lab}: {pos_workspace[i]:.1f} mm",
                        (x, y + 14 * (i + 1)),
                        font, 0.28, (0, 255, 255), 1
                    )

        else:
            cv2.putText(color_frame, "No tag detected",
                        (10, 60), font, 0.7, (0, 0, 255), 2)

        # ---- FPS overlay ----
        cv2.putText(color_frame, f"FPS: {fps:.1f}",
                    (color_frame.shape[1] - 140, 30),
                    font, 0.7, (0, 255, 0), 2)

        cv2.putText(color_frame, "Press 'q' to quit",
                    (10, 30), font, 0.7, (0, 255, 0), 2)

        cv2.imshow("Calibration Validation", color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---- Cleanup ----
    cap.release()
    cv2.destroyAllWindows()

    data_time = data_time[:count]
    data_ee_pos = data_ee_pos[:count]

    save_to_pickle({
        "time": data_time,
        "pos_current": data_ee_pos
    }, "Camera_Tag_Tracking.pkl")

    print("Recording complete.")

if __name__ == "__main__":
    print("ENTERING MAIN")
    record()
    print("RECORD RETURNED")
    #ploting()

