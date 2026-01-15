"""
(c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
Lab 6 Starter Code: AprilTag Detection and Pose Estimation Test Script
Tests the integration of RealSense camera with AprilTag detection and pose estimation.
"""
import numpy as np
import cv2
from time import time
#from classes.Realsense import Realsense
from AprilTags import AprilTags
import matplotlib.pyplot as plt

class Intrinsics:
    def __init__(self,fx,fy,ppx,ppy):
        self.fx = fx
        self.fy=fy
        self.ppx=ppx
        self.ppy=ppy

def plot(logs):
    times, xs, ys = zip(*logs)
    plt.figure()
    plt.plot(xs, ys, '-o')
    plt.title("2D Agent 0 Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    logs = []
    #cap = cv2.VideoCapture(0)
    try:
        # Get camera intrinsic parameters
        intrinsics = Intrinsics(1400,1400,960,540)

        # Initialize AprilTag detector
        at = AprilTags()

        # Tag size in millimeters (measure your actual tag size)
        TAG_SIZE = 100.0  # 40mm tag (update this to match your actual tag size)

        # Counter for controlling print frequency
        counter = 0
        last_time = time()

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
        print("FOURCC:", "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)]))
        print("test")

        while True:
            # Get frames from RealSense
            ret, frame = cap.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            cv2.imshow("MSMF Camera", gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Detect AprilTags
            tags = at.detect_tags(gray)

            # Process each detected tag
            for tag in tags:
                # Draw tag detection on image
                at.draw_tags(gray, tag)

                # Get pose estimation
                rot_matrix, trans_vector = at.get_tag_pose(
                    tag.corners,
                    intrinsics,
                    TAG_SIZE)
                # Print every 10 frames (approximately)
                if counter % 10 == 0:
                    if rot_matrix is not None and trans_vector is not None:
                        # Convert rotation matrix to Euler angles
                        euler_angles = cv2.RQDecomp3x3(rot_matrix)[0]

                        # Calculate distance (norm of translation vector)
                        distance = np.linalg.norm(trans_vector)

                        # Print results
                        print(f"\nTag ID: {tag.tag_id}")
                        print(f"Distance: {distance:.1f} mm")
                        print(f"Orientation (deg): roll={euler_angles[0]:.1f}, "
                              f"pitch={euler_angles[1]:.1f}, "
                              f"yaw={euler_angles[2]:.1f}")
                        print(f"Position (mm): x={trans_vector[0][0]:.1f}, "
                              f"y={trans_vector[1][0]:.1f}, "
                              f"z={trans_vector[2][0]:.1f}")
                        # Calculate and print frame rate
                        current_time = time()

                        # Append log data
                        logs.append([current_time,trans_vector[0][0], trans_vector[1][0]])

                        # fps = 1.0 / (current_time - last_time)
                        # print(f"FPS: {fps:.1f}")
                        last_time = current_time

            # Display the image
            cv2.imshow('AprilTag Detection', gray)

            # Increment counter
            counter += 1

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                plot(logs)
                time.sleep(1)
                while(1):
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except Exception as e:
        print(f"Error in main loop: {str(e)}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
