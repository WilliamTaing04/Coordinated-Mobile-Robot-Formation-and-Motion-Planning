import cv2
import AprilTags

def setup():
    detector = AprilTags.AprilTags()
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not cap.isOpened():
        print("Failed to open camera with MSMF")
        exit()
    return detector, cap

detector, cap=setup()
i = 500
while True:
    
    cap.set(cv2.CAP_PROP_FOCUS, i)
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame,f"Focus level = {i}", (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    tags = detector.detect_tags(frame)  # Replace with detected tags
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
    elif key == ord('s'):
        i -= 1
cap.release()
cv2.destroyAllWindows()
