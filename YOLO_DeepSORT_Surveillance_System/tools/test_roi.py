import cv2

def test_roi(video_path, frame_idx=0):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "video open failed"

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("read frame failed")
        return

    # ===== 选 car =====
    print("Select car ROI, then press ENTER")
    car = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)

    # ===== 选 door =====
    print("Select door ROI, then press ENTER")
    door = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)

    cv2.destroyAlldoors()

    def to_xyxy(roi):
        x, y, w, h = roi
        return (int(x), int(y), int(x + w), int(y + h))

    car_xyxy = to_xyxy(car)
    door_xyxy = to_xyxy(door)

    print("\n===== COPY THESE INTO YOUR CODE =====")
    print(f"car_roi   = {car_xyxy}")
    print(f"door_roi = {door_xyxy}")

test_roi("miss.mp4", frame_idx=300)
