import os
import glob
import cv2
import numpy as np
import json
import csv
from ultralytics import YOLO
from src.pdf_to_images import PDFConverter
from src.superglue_aligner import DeepLearningAligner
from src.gap_analysis import Comparator

# --- Moved from scripts/yolo_detector.py ---
class YOLODetector:
    # Using Hugging Face handwritten text detection model
    # UPDATED: Added min_box_area to filter out tiny noise detections
    def __init__(self, model_path=None, conf_thres=0.15, iou_thres=0.45, imgsz=1280, min_box_area=300):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.min_box_area = min_box_area
        self.model = self._load_model()

    def _load_model(self):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_local_path = os.path.join(base_dir, "weights", "best.pt")

            if self.model_path:
                chosen_path = self.model_path
                if not os.path.isabs(chosen_path):
                    chosen_path = os.path.join(base_dir, chosen_path)
                if os.path.exists(chosen_path):
                    print(f"Loading local model: {chosen_path}")
                    return YOLO(chosen_path)
                raise FileNotFoundError(f"Configured model path does not exist: {chosen_path}")

            if os.path.exists(default_local_path):
                print(f"Loading local model: {default_local_path}")
                return YOLO(default_local_path)

            raise FileNotFoundError(
                f"Local model not found at {default_local_path}. Place best.pt in the weights folder."
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def detect(self, image):
        """
        Run inference on a single image (numpy array or path).
        Returns a list of detections. containing class, bbox, conf.
        """
        results = self.model.predict(
           source=image, 
           conf=self.conf_thres, 
           iou=self.iou_thres,
           imgsz=self.imgsz,
           augment=True, 
           verbose=False
        )
        
        detections = []
        for result in results:
            for box in result.boxes: # type: ignore
                # box.xyxy is [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                
                # Check Box Area
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # FILTER: If box is too small (noise), skip it
                if area < self.min_box_area:
                    continue

                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                label = result.names[cls]
                
                # We specifically want to catch handwriting or relevant text fields
                # If the model has specific classes for handwriting, filter here.
                # For now, we return all detections.
                
                detections.append({
                    "label": label,
                    "box": coords.tolist(), # [x1, y1, x2, y2]
                    "confidence": conf
                })
        return detections

# --- Moved from scripts/output_handler.py ---
class OutputHandler:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_json(self, data, filename):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved JSON report to {path}")

    def save_config(self, config, filename="config_summary.json"):
        """Save run configuration parameters"""
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved config summary to {path}")

    def save_csv(self, data, filename):
        path = os.path.join(self.output_dir, filename)
        if not data:
            return
            
        keys = data[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved CSV report to {path}")

    def visualize_missing(self, image, missing_detections, filename, page_num):
        """
        Draw bounding boxes of missing data on the test image (or aligned test image).
        """
        vis_img = image.copy()
        
        for item in missing_detections:
            box = item['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Draw Red Rectangle for Missing
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_img, f"Missing {item.get('label','Data')}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        output_path = os.path.join(self.output_dir, f"{filename}_page_{page_num}_missing_vis.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"Saved visualization to {output_path}")

    def visualize_detections(self, image, detections, filename, page_num):
        """
        Draw bounding boxes of all YOLO detections on the image.
        """
        vis_img = image.copy()
        
        for item in detections:
            box = item['box']
            x1, y1, x2, y2 = map(int, box)
            conf = item.get('confidence', 0)
            label = item.get('label', 'detected')
            
            # Draw Green Rectangle for all detections
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(self.output_dir, f"{filename}_page_{page_num}_detections.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"Saved YOLO detections visualization to {output_path}")
        return vis_img

    def visualize_alignment(self, ref_img, test_img, matches_info, filename, page_num):
        """
        Visualize alignment quality with matched keypoints and metrics.
        """
        # Create side-by-side visualization
        h1, w1 = ref_img.shape[:2]
        h2, w2 = test_img.shape[:2]
        
        # Create canvas
        max_h = max(h1, h2)
        vis_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if len(ref_img.shape) == 2 else ref_img
        vis_img[:h2, w1:w1+w2] = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR) if len(test_img.shape) == 2 else test_img
        
        # Draw matches
        kpts0 = matches_info.get('keypoints0', np.array([]))
        kpts1 = matches_info.get('keypoints1', np.array([]))
        matches = matches_info.get('matches', np.array([]))
        confidence = matches_info.get('confidence', np.array([]))
        
        num_matches = len(matches)
        
        # Draw top 50 best matches (sorted by confidence)
        if num_matches > 0 and len(confidence) > 0:
            sorted_indices = np.argsort(confidence)[::-1][:50]
            
            for idx in sorted_indices:
                if idx < len(matches) and matches[idx][1] >= 0:
                    pt1 = tuple(map(int, kpts0[matches[idx][0]]))
                    pt2 = tuple(map(int, kpts1[matches[idx][1]]))
                    pt2_shifted = (pt2[0] + w1, pt2[1])
                    
                    # Color based on confidence
                    conf = confidence[idx]
                    if conf > 0.8:
                        color = (0, 255, 0)  # Green
                    elif conf > 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 165, 255)  # Orange
                    
                    cv2.line(vis_img, pt1, pt2_shifted, color, 1)
                    cv2.circle(vis_img, pt1, 3, (255, 0, 0), -1)
                    cv2.circle(vis_img, pt2_shifted, 3, (0, 0, 255), -1)
        
        # Add text overlay with statistics
        inlier_ratio = matches_info.get('inlier_ratio', 0)
        avg_conf = np.mean(confidence) if len(confidence) > 0 else 0
        
        stats_text = [
            f"Matches: {num_matches}",
            f"Inlier Ratio: {inlier_ratio:.2%}",
            f"Avg Confidence: {avg_conf:.3f}",
            f"Quality: {'EXCELLENT' if num_matches > 100 else 'GOOD' if num_matches > 50 else 'FAIR' if num_matches > 20 else 'POOR'}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(vis_img, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 40
        
        output_path = os.path.join(self.output_dir, f"{filename}_page_{page_num}_alignment.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"Saved alignment visualization to {output_path}")
        print(f"  → {num_matches} matches, {inlier_ratio:.1%} inliers, {avg_conf:.3f} avg confidence")

    def save_debug_image(self, image, filename):
        """Save debug image to output directory"""
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, image)

# --- Moved from scripts/aligner.py ---
# This class is now superseded by the DeepLearningAligner from lightglue_aligner.py
# Keeping this commented for reference or fallback if needed.
# class ImageAligner:
#     ... (Legacy SIFT Implementation) ...




# The model 'armvectores/yolov8n_handwritten_text_detection' returns class 'word'.
# We will accept 'word', 'handwritten', 'handwriting', or just return everything if broad.
HANDWRITING_LABEL_KEYWORDS = ["word", "handwritten", "handwriting", "text"]


def get_first_pdf(folder):
    pdfs = glob.glob(os.path.join(folder, "*.pdf"))
    return pdfs[0] if pdfs else None


def filter_handwriting(detections):
    if not HANDWRITING_LABEL_KEYWORDS:
        return detections
    filtered = []
    for d in detections:
        label = d.get("label", "").lower()
        if any(k in label for k in HANDWRITING_LABEL_KEYWORDS):
            filtered.append(d)
    return filtered


def main():
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION: Tweak these parameters to improve detection accuracy
    # ═══════════════════════════════════════════════════════════════════
    
    # YOLO Detection Parameters
    CONF_THRESHOLD = 0.0001   # Confidence (0.0-1.0): Higher = fewer false positives, lower = catches more faint text
    NMS_IOU_THRESHOLD = 0.0001    # NMS IoU (0.0-1.0): Higher = more duplicate boxes, lower = cleaner results
    IMAGE_SIZE = 2560            # Image resolution: Higher = better for small text but slower
    MIN_BOX_AREA = 700      # Minimum box area (pixels): Filter out tiny noise detections
    
    # Comparison Parameters
    COMPARATOR_IOU = 0.10     # Match threshold (0.0-1.0): Lower = more forgiving, higher = stricter matching
    
    # Alignment Parameters
    MAX_KEYPOINTS = 8192         # Feature points: Higher = better alignment but slower
    
    # ═══════════════════════════════════════════════════════════════════
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(base_dir, "reference_record")
    test_dir = os.path.join(base_dir, "test_record")

    ref_pdf = get_first_pdf(ref_dir)
    test_pdf = get_first_pdf(test_dir)

    if not ref_pdf or not test_pdf:
        print("Error: Could not find PDF in reference_record or test_record.")
        return

    print(f"Reference: {ref_pdf}")
    print(f"Test: {test_pdf}")

    pdf_converter = PDFConverter()
    ref_images = pdf_converter.convert_to_images(ref_pdf)
    test_images = pdf_converter.convert_to_images(test_pdf)

    if not ref_images or not test_images:
        print("No images converted.")
        return

    print("Loading YOLO Model...")
    detector = YOLODetector(
        conf_thres=CONF_THRESHOLD,
        iou_thres=NMS_IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        min_box_area=MIN_BOX_AREA
    )
    
    print("Loading AI Aligner (SuperPoint+LightGlue)...")
    aligner = DeepLearningAligner(max_keypoints=MAX_KEYPOINTS)
    
    comparator = Comparator(iou_threshold=COMPARATOR_IOU)
    output_handler = OutputHandler(output_dir=os.path.join(base_dir, "output_results"))

    all_missing_data = []
    all_ref_detections = []
    all_test_detections = []

    min_pages = min(len(ref_images), len(test_images))
    print(f"Processing {min_pages} pages...")

    for i in range(min_pages):  # Process only first page
        if i != 1:
                continue  # Skip all but page 2 for now
        print(f"--- Processing Page {i+1} ---")
        ref_img = ref_images[i]
        test_img = test_images[i]

        print("Aligning Test Image to Reference...")
        aligned_test_img, H, matches_info = aligner.align(test_img, ref_img)
        output_handler.save_debug_image(aligned_test_img, f"debug_aligned_page_{i+1}.jpg")
        
        # Visualize alignment quality
        output_handler.visualize_alignment(ref_img, test_img, matches_info, "alignment", i + 1)

        print("Detecting on Reference...")
        ref_detections = detector.detect(ref_img)
        print(f"Found {len(ref_detections)} total items in Reference.")
        output_handler.visualize_detections(ref_img, ref_detections, "reference", i + 1)

        print("Detecting on Test...")
        test_detections = detector.detect(aligned_test_img) # Use aligned image
        print(f"Found {len(test_detections)} total items in Test.")
        output_handler.visualize_detections(aligned_test_img, test_detections, "test", i + 1)

        for item in ref_detections:
            item['page'] = i + 1
            item['file'] = os.path.basename(ref_pdf)
            item['source'] = 'reference'
            all_ref_detections.append(item)

        for item in test_detections:
            item['page'] = i + 1
            item['file'] = os.path.basename(test_pdf)
            item['source'] = 'test'
            all_test_detections.append(item)

        print("Comparing...")
        missing_on_page = comparator.find_missing_data(ref_detections, test_detections)
        print(f"Found {len(missing_on_page)} missing items.")

        for item in missing_on_page:
            item['page'] = i + 1
            item['file'] = os.path.basename(test_pdf)
            all_missing_data.append(item)

        output_handler.visualize_missing(aligned_test_img, missing_on_page, "comparison", i + 1)

    #output_handler.save_json(all_ref_detections, "reference_handwriting_boxes.json")
    #output_handler.save_json(all_test_detections, "test_handwriting_boxes.json")
    #output_handler.save_json(all_missing_data, "missing_data_report.json")

    #output_handler.save_csv(all_ref_detections, "reference_handwriting_boxes.csv")
    #output_handler.save_csv(all_test_detections, "test_handwriting_boxes.csv")
    #output_handler.save_csv(all_missing_data, "missing_data_report.csv")
    # Save configuration summary
    import torch
    config_summary = {
        "yolo_parameters": {
            "confidence_threshold": CONF_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "image_size": IMAGE_SIZE,
            "min_box_area": MIN_BOX_AREA,
            "model": "armvectores/yolov8n_handwritten_text_detection"
        },
        "alignment_parameters": {
            "method": "SuperPoint + LightGlue",
            "max_keypoints": MAX_KEYPOINTS,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "comparison_parameters": {
            "comparator_iou_threshold": COMPARATOR_IOU
        }
    }
    output_handler.save_config(config_summary)
    print("Done.")


if __name__ == "__main__":
    main()
