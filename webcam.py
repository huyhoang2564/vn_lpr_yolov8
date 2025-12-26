import cv2
import time
from ultralytics import YOLO
import function.utils_rotate as utils_rotate
import function.helper as helper

# =========================
# CONFIG
# =========================
LP_MODEL_PATH = r"model/best_lp.pt"
CHAR_MODEL_PATH = r"model/best_char.pt"

IMG_SIZE = 640
PLATE_CONF = 0.25

# =========================
# Camera helper
# =========================
def find_camera(max_index=5):
    """Try to find a working camera index on Windows."""
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return None


def main():
    # ---- Load models ----
    try:
        yolo_LP_detect = YOLO(LP_MODEL_PATH)
        yolo_license_plate = YOLO(CHAR_MODEL_PATH)
    except Exception as e:
        print("❌ Lỗi load model! Kiểm tra đường dẫn model hoặc file .pt.")
        print(e)
        return

    # ---- Open camera ----
    cam_idx = find_camera(5)
    if cam_idx is None:
        print("❌ Không tìm thấy camera nào (0..5).")
        print("👉 Kiểm tra: Windows Camera permission + đóng Zalo/Teams/Browser đang dùng cam.")
        return

    print(f"✅ Using camera index: {cam_idx}")
    vid = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)

    if not vid.isOpened():
        print("❌ Mở camera thất bại.")
        return

    prev_time = time.time()

    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            print("⚠️ Không đọc được frame từ camera.")
            break

        # ---- YOLO plate detect ----
        plates_results = yolo_LP_detect(frame, imgsz=IMG_SIZE, conf=PLATE_CONF, verbose=False)[0]
        list_plates = plates_results.boxes.data.tolist() if plates_results.boxes is not None else []

        for plate in list_plates:
            x1, y1, x2, y2 = map(int, plate[:4])

            # clamp to image bounds
            h_img, w_img = frame.shape[:2]
            x1 = max(0, min(x1, w_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            y2 = max(0, min(y2, h_img - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop_img = frame[y1:y2, x1:x2].copy()

            # draw plate box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ---- Read plate by trying deskew params ----
            lp_text = "unknown"
            found = False
            for cc in range(0, 2):
                for ct in range(0, 2):
                    rotated = utils_rotate.deskew(crop_img, cc, ct)
                    lp_text = helper.read_plate(yolo_license_plate, rotated)
                    if lp_text != "unknown":
                        found = True
                        break
                if found:
                    break

            if lp_text != "unknown":
                cv2.putText(frame, lp_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ---- FPS ----
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0 if dt <= 0 else 1.0 / dt
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = f"debug_frame_{int(time.time())}.jpg"
            cv2.imwrite(out_path, frame)
            print("✅ Saved:", out_path)

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
