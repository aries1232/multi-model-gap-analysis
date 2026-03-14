from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
 
# Load the weights from our repository
model_path = hf_hub_download(local_dir=".",
                             repo_id="armvectores/yolov8n_handwritten_text_detection",
                             filename="best.pt")
model = YOLO(model_path)
 
# Load and resize image to 1604x2200 (width x height)
img = cv2.imread(r"D:\zippaidocs\batch_record_research2\scripts\cropped\reference_page_2.png")
img = cv2.resize(img, (1600, 550))

# 50 combinations of (conf, iou) to try
# conf from 0.05 to 0.50 in steps of 0.05  -> 10 values
# iou  from 0.30 to 0.70 in steps of 0.10  -> 5 values
# total = 10 * 5 = 50 combinations
conf_values = [0.05 + 0.05 * i for i in range(10)]  # [0.05, 0.10, ..., 0.50]
iou_values = [0.3 + 0.1 * i for i in range(5)]      # [0.30, 0.40, ..., 0.70]

conf_iou_combinations = [
    (round(conf, 2), round(iou, 2))
    for conf in conf_values
    for iou in iou_values
]
 
# Run prediction for each combination and save
for conf, iou in conf_iou_combinations:
    run_name = f"conf{conf}_iou{iou}".replace(".", "p")
    print(f"Running: conf={conf}, iou={iou} -> {run_name}")
    res = model.predict(
        source=img,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        save=True,
        show=False,
        show_labels=False,
        show_conf=False,
        conf=conf,
        iou=iou,
        imgsz=(550, 1600),  # (height, width) to match resized image
        augment=True,
        verbose=False,
    )
    n_det = len(res[0].boxes) if res[0].boxes is not None else 0
    print(f"  -> {n_det} detections saved to runs/detect/{run_name}/")
 
print("\nDone! All 10 runs saved under runs/detect/")