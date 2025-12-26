from PIL import Image
import cv2
from ultralytics import YOLO # Thay torch bằng ultralytics
import math 
import function.utils_rotate as utils_rotate
from IPython.display import display
import os
import time
import argparse
import function.helper as helper

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

# --- SỬA LOAD MODEL ---
# Lưu ý: Bạn cần có file model .pt tương thích (train bằng v8 hoặc v5 load được)
# Nếu bạn chưa train lại, hãy thử dùng file cũ xem có load được không
try:
    yolo_LP_detect = YOLO('model/best_lp.pt') 
    yolo_license_plate = YOLO('model/best_char.pt')
except Exception as e:
    print("Lỗi load model! Bạn cần train lại model bằng YOLOv8 hoặc đảm bảo file .pt tương thích.")
    print(e)
    exit()

# Set confidence nếu cần (với v8 thì truyền lúc predict)
# yolo_license_plate.conf = 0.60 

img = cv2.imread(args.image)

# --- SỬA INFERENCE ---
# detect biển số
plates_results = yolo_LP_detect(img, imgsz=640, conf=0.25) # conf mặc định

# Lấy list các biển số phát hiện được
list_plates = plates_results[0].boxes.data.tolist()

list_read_plates = set()
if len(list_plates) == 0:
    # Nếu không tìm thấy biển, thử detect ký tự trên toàn ảnh (fallback)
    lp = helper.read_plate(yolo_license_plate, img)
    if lp != "unknown":
        cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        list_read_plates.add(lp)
else:
    for plate in list_plates:
        flag = 0
        x = int(plate[0]) # xmin
        y = int(plate[1]) # ymin
        w = int(plate[2] - plate[0]) # xmax - xmin -> w
        h = int(plate[3] - plate[1]) # ymax - ymin -> h
        
        # Crop ảnh
        crop_img = img[y:y+h, x:x+w]
        cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
        
        cv2.imwrite("crop.jpg", crop_img)
        # rc_image = cv2.imread("crop.jpg") # Dòng này thừa

        lp = ""
        for cc in range(0,2):
            for ct in range(0,2):
                # Deskew và nhận diện
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    flag = 1
                    break
            if flag == 1:
                break

cv2.imshow('frame', img)
cv2.waitKey()
cv2.destroyAllWindows()