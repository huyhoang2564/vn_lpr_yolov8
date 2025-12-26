# Vietnamese License Plate Recognition (YOLOv8)

A Vietnamese motorcycle and car license plate recognition system using **YOLOv8**, optimized to run on both **CPU** and **GPU**. The system operates in 2 stages (2-stage detection):
1.  **License Plate Detection:** Identifies the location of license plates in the frame.
2.  **Character Recognition:** Crops, rotates (deskew) the license plate, and reads the characters.

---

## Prerequisites

*   **OS:** Windows 10/11 (Recommended), Linux, MacOS.
*   **Python:** 3.8 - 3.11 (Recommended 3.10 or 3.11).
*   **GPU (Optional):** NVIDIA GPU with CUDA support for real-time processing speed.

---

## Installation

### 1. Clone the project
Download the project to your computer:
```bash
git clone <YOUR_GITHUB_LINK>
cd vn_lpr_yolov8
```

### 2. Create virtual environment (Recommended)
Helps avoid library conflicts with the main system:
```bash
python -m venv .venv
# Activate the environment:
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install basic libraries
Run the following command to install necessary libraries (`ultralytics`, `opencv`, `numpy`...):
```bash
pip install -r requirement.txt
```

### 4. Install PyTorch with GPU support (Important)
For smooth system operation and fast training, you should install the CUDA-supported PyTorch version.
*   **Step 1:** Check your machine's CUDA version (if you have an NVIDIA card).
*   **Step 2:** Run the corresponding installation command (Example for CUDA 12.1):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(If using CPU only, you can skip this step, but speed will be slower)*

---

## Usage

The project includes 2 pre-trained best models in the `model/` directory:
*   `best_lp.pt`: License plate detection model.
*   `best_char.pt`: Character recognition model.

### 1. Image Inference
To test on any image file:
```bash
python lp_image.py -i test_images/car1.jpg
# Or replace with your image path:
python lp_image.py -i "D:/photos/my_car.jpg"
```
*Results will be displayed on screen with bounding boxes drawn on the image.*

### 2. Webcam Real-time Detection
To run real-time recognition using camera (Webcam):
```bash
python webcam.py
```
*   The system will automatically find available cameras.
*   Press **`s`** to save the current frame.
*   Press **`q`** to exit the program.

---

## Re-training Models

If you have a new dataset and want to retrain the models:

1.  **Prepare data:** Place dataset in `dataset/` directory following standard YOLO structure.
2.  **Configuration:** Edit `training/LP_detection.yaml` (license plate) and `training/Letter_detect.yaml` (character) files if needed.
3.  **Run training:** Use the automated script to train both models sequentially:
```bash
python training/train_all.py
```
*This script is configured to automatically use GPU, Early Stopping, and save the best results.*

---

## Project Structure

```text
vn_lpr_yolov8/
├── model/                  # Contains best weights (.pt) files
│   ├── best_lp.pt          # License plate model
│   └── best_char.pt        # Character model
├── function/               # Helper processing functions
│   ├── helper.py           # Character reading and sorting logic
│   └── utils_rotate.py     # Image rotation and deskewing
├── training/               # Training scripts
├── test_images/            # Sample test images
├── webcam.py               # Real-time detection script
├── lp_image.py             # Image detection script
└── requirement.txt         # Library dependencies
```

---

## Troubleshooting

**Camera not detected:**
- Ensure camera permissions are enabled in Windows Settings
- Close other applications using the camera (Zalo, Teams, Browser)
- Try changing camera index in `webcam.py`

**CUDA/GPU not working:**
- Verify CUDA installation: `nvidia-smi`
- Ensure PyTorch CUDA version matches your CUDA version
- Check GPU availability in Python: `torch.cuda.is_available()`

---

## Credits
This project is developed based on **Ultralytics YOLOv8** framework.
