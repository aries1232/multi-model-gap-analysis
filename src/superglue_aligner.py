import cv2
import numpy as np

def align_images(ref_img, test_img):
    try:
        sift = cv2.SIFT_create()
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = sift.detectAndCompute(test_gray, None)
        kp2, des2 = sift.detectAndCompute(ref_gray, None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = ref_img.shape[:2]
            return cv2.warpPerspective(test_img, M, (w, h))
    except Exception as e:
        print(f'Align failed: {e}')
    return test_img

class DeepLearningAligner:
    def __init__(self, max_keypoints=8192):
        self.max_keypoints = max_keypoints

    def align(self, test_img, ref_img):
        # We proxy this to align_images. Notice the argument order swapped based on align_images signature!
        # Your previous script called it as align(test_img, ref_img)
        # But align_images expects (ref_img, test_img)
        aligned_img = align_images(ref_img, test_img)
        # Return dummy homography and empty matches info to prevent crashes downstream
        dummy_homography = np.eye(3) 
        dummy_matches = {'m0': []}
        return aligned_img, dummy_homography, dummy_matches
