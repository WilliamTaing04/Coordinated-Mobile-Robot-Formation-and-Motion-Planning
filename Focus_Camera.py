import cv2
import time
import AprilTags

def setup():
    detector = AprilTags.AprilTags()
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode (driver dependent)
    # cap.set(cv2.CAP_PROP_EXPOSURE, 100)        # smaller values → shorter exposure
    # cap.set(cv2.CAP_PROP_GAIN, 0)              # adjust to taste


    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600) 

    cap.set(cv2.CAP_PROP_FPS, 100)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution: {int(width)} x {int(height)}")

    if not cap.isOpened():
        print("Failed to open camera with MSMF")
        exit()
    return detector, cap

detector, cap=setup()
i = 500
prev = time.perf_counter()

frames = 1000
count = 0
start = time.perf_counter()

while count < frames:
    start_time = time.perf_counter()
    count += 1
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    # cv2.putText(frame,f"Focus level = {i}", (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    tags = detector.detect_tags(frame)  # Replace with detected tags
    # if len(tags) > 0:
        # print(f"hz: {1.0/(time.perf_counter()-start_time)}")

    for tag in tags:
        detector.draw_tags(frame, tag)
        #y_offset = 60
        # cv2.putText(frame, f"Tag ID: {tag.tag_id}",
        #     (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6, (0, 255, 0), 2)

    cv2.imshow("MSMF Camera", frame)

    
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"Optimal focus level is ", i)
        break
    elif key == ord('w'):
        i += 1
        cap.set(cv2.CAP_PROP_FOCUS, i)
    elif key == ord('s'):
        i -= 1
        cap.set(cv2.CAP_PROP_FOCUS, i)
    elif key == ord('e'):
        i += 20
        cap.set(cv2.CAP_PROP_FOCUS, i)
    elif key == ord('d'):
        i -= 20
        cap.set(cv2.CAP_PROP_FOCUS, i)
    elif key == ord('r'):
        h, w = frame.shape[:2]
        print(f"Frame resolution: {w} x {h}")

end = time.perf_counter()
print("Avg fps:", count / (end - start))

cap.release()
cv2.destroyAllWindows()
