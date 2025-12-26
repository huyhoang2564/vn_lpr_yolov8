# training/train_letter.py
from __future__ import annotations

import os
from multiprocessing import freeze_support
from pathlib import Path

import torch
from ultralytics import YOLO

# Tắt torch.compile / TorchDynamo / Inductor (giảm treo/trace dài trên Windows)
os.environ.setdefault("TORCH_COMPILE", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")


def main() -> None:
    data_yaml = Path(r"E:/VNUK_Project/vn_lpr_yolov8/training/Letter_detect.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy file YAML: {data_yaml}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA không khả dụng. Hãy kiểm tra torch CUDA trong đúng .venv.")
    print("✅ GPU:", torch.cuda.get_device_name(0))

    # Load pretrained
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=str(data_yaml),

        # ===== CORE =====
        epochs=120,
        patience=25,
        imgsz=832,          # RTX 2050 4GB: 832 thường ổn hơn 960; nếu OK có thể lên 960
        batch=4,
        device=0,
        workers=2,          # nếu còn lỗi multiprocessing -> set 0
        amp=True,
        pretrained=True,

        # ===== AUGMENT (hợp lệ YOLOv8) =====
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,

        degrees=2.0,
        translate=0.05,
        scale=0.30,
        shear=0.0,
        perspective=0.0,

        # màu/ánh sáng (giúp biển nền xanh)
        hsv_h=0.03,
        hsv_s=0.60,
        hsv_v=0.50,

        # erasing là arg hợp lệ (rand erasing)
        erasing=0.20,

        # multi-scale có thể làm chữ quá nhỏ -> tắt để ổn định
        multi_scale=False,

        # ===== OPT/STABILITY =====
        optimizer="auto",
        lr0=0.003,
        warmup_epochs=3.0,
        weight_decay=0.0005,

        compile=False,
        cache=False,

        # ===== SAVE =====
        project=r"runs/detect",
        name="letter_train_y8s_832",
        exist_ok=True,
        save_period=1,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    print("\n Train xong!")
    print(" Save dir:", save_dir)
    print(" best.pt:", save_dir / "weights" / "best.pt")
    print(" last.pt:", save_dir / "weights" / "last.pt")


if __name__ == "__main__":
    freeze_support()
    main()
