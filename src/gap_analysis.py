class Comparator:
    # Lowered IoU threshold to 0.1 to account for lack of alignment.
    # If a box in Ref overlaps even 10% with a box in Test, we consider it "Present".
    def __init__(self, iou_threshold=0.1):
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        Box format: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0
        
        return intersection_area / union_area

    def find_missing_data(self, ref_detections, test_detections):
        """
        Identify detections present in Reference but missing in Test.
        Returns a list of missing detections (from reference).
        """
        missing_items = []

        for ref_item in ref_detections:
            ref_box = ref_item['box']
            match_found = False
            
            for test_item in test_detections:
                test_box = test_item['box']
                iou = self.calculate_iou(ref_box, test_box)
                
                if iou > self.iou_threshold:
                    match_found = True
                    break
            
            if not match_found:
                # Add metadata to the missing item
                item_copy = ref_item.copy()
                item_copy['status'] = "MISSING_IN_TEST"
                missing_items.append(item_copy)

        return missing_items
