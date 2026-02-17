import cv2
import time
import AprilTags

def setup():
    detector = AprilTags.AprilTags()
    # cap = cv2.VideoCapture(1, cv2.CAP_MSMF)    
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # switch to DirectShow

    # Prefer MJPG for high FPS on Arducam UVC modules
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # CHOSEN 1280, 720
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600) 
    cap.set(cv2.CAP_PROP_FPS, 200)   # try 90 or 100 (module advertises ~90–100fps in MJPG)

    # Latency / buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Lock exposure/focus (DSHOW + this camera usually respect these)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # often 0.25 = manual, 0.75 = auto (driver-dependent)
    # set a short exposure (value is camera/driver-dependent; try negative or small positive)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # Immediately read back what the driver actually set
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Actual: {w}x{h} @ {fps:.1f} FPS")

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera (DSHOW)")



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

frames = 200000
count = 0
at_count = 0
start = time.perf_counter()

while count < frames:
    start_time = time.perf_counter()
    count += 1
    ret, frame = cap.retrieve()
    if not ret:
        break

    now = time.perf_counter()
    fps = 1 / (now - prev)
    prev = now

    tags = detector.detect_tags(frame)  # Replace with detected tags
    if len(tags) > 0:
        at_count += 1
    # if len(tags) > 0:
        # print(f"hz: {1.0/(time.perf_counter()-start_time)}")

    for tag in tags:
        detector.draw_tags(frame, tag)
        #y_offset = 60
        # cv2.putText(frame, f"Tag ID: {tag.tag_id}",
        #     (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    # cv2.putText(frame,f"Focus level = {i}", (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    # if count % 3 == 0:
    #     for tag in tags:
    #         detector.draw_tags(frame, tag)
    #         #y_offset = 60
    #         # cv2.putText(frame, f"Tag ID: {tag.tag_id}",
    #         #     (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
    #         #     0.6, (0, 255, 0), 2)

    #     cv2.putText(frame, f"FPS: {fps:.1f}",
    #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (0, 255, 0), 2)
    #     # cv2.putText(frame,f"Focus level = {i}", (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
    #     cv2.imshow("MSMF Camera", frame)
    #     cv2.waitKey(1)

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
print(f"AT count {at_count}     {100*(at_count/count)}%")

cap.release()
cv2.destroyAllWindows()
