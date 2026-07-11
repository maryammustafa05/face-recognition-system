import cv2

for index in [0, 1, 2]:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera {index}: {'✅ Working' if ret else '❌ Opens but no frame'}")
        cap.release()
    else:
        print(f"Camera {index}: ❌ Not found")