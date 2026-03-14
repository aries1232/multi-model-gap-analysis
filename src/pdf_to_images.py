import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

class PDFConverter:
    def __init__(self, dpi=300):
        self.dpi = dpi
        
    def convert_to_images(self, pdf_path):
        """
        Converts a PDF file into a list of OpenCV images (numpy arrays).
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=self.dpi)
                
                # Convert the pixmap to a PIL Image
                if pix.alpha:
                    img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
                    img = img.convert("RGB")
                else:
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert PIL Image to OpenCV BGR format
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img_cv)
                
            return images
        except Exception as e:
            print(f"Error converting PDF {pdf_path} to images: {e}")
            return []
