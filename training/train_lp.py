# training/train_lp.py
from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    # Path tới lp.yaml
    data_yaml = Path(__file__).resolve().parent / "LP_detection.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy {data_yaml}")

    # Kiểm tra CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng. Kiểm tra lại torch CUDA trong .venv.")

    print("✅ GPU:", torch.cuda.get_device_name(0))

    # Load pretrained model
    model = YOLO("yolov8n.pt")

    # Train và LƯU VÀO runs/detect
    results = model.train(
        data=str(data_yaml),
        epochs=40,
        patience=10,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,              # nếu lỗi multiprocessing -> đổi 0
        project="runs/detect",  # <<< QUAN TRỌNG
        name="lp",              # thư mục con: runs/detect/lp
        exist_ok=False,         # False -> lp, lp2, lp3...
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("\n Train xong!")
    print(" Runs dir:", save_dir)
    print(" Best weights:", save_dir / "weights" / "best.pt")
    print(" Last weights:", save_dir / "weights" / "last.pt")


if __name__ == "__main__":
    freeze_support()   # BẮT BUỘC trên Windows
    main()
