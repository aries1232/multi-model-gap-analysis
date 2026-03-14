# Document Intelligence Pipeline

An automated solution for document alignment and handwriting detection using state-of-the-art computer vision models.

## 🚀 Key Features

- **Document Alignment**: Uses SuperPoint and LightGlue for robust image alignment between test and reference documents.
- **Handwriting Detection**: Leverages YOLO-based models (YOLOv11, YOLOv8) to identify handwritten regions within documents.
- **Support for Large Documents**: Integrated PDF processing to handle multi-page document comparisons.

## 📁 Project Structure

```text
├── weights/           # Pre-trained YOLO and Feature Extraction weights
├── src/               # Core logic (field_identification.py, lightglue_aligner.py, gap_analysis.py, parameter_tuning.py)
├── data/              # Input/Output data (PDFs, Images, Results)
├── main.py            # Main entry point to run the pipeline
├── requirements.txt   # Project dependencies
└── README.md          # Documentation
```

## 🚀 Key Modules

The pipeline follow a multi-stage automated workflow within the `src/` directory:

1. **Field Identification**: Uses YOLOv8/v11 for detecting handwritten fields within documents.
2. **Document Alignment**: Uses SuperPoint + LightGlue to align test documents to a reference master.
3. **Gap Analysis**: Compares the identified fields of aligned documents to find deviations.
4. **Parameter Tuning**: Integrated tools for optimizing YOLO model confidence and overlap thresholds.

## 🛠️ Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd document-intelligence-pipeline
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **External Requirements:**
   - [Optional] CUDA for GPU acceleration on alignment.

## 🚀 Usage

Run the main pipeline:

```bash
python main.py
```

## 🧠 Technologies Used

- **Deep Learning**: YOLOv11, YOLOv8, LightGlue, SuperPoint
- **Core Vision**: PyTorch, OpenCV, Pillow
- **Data Engineering**: PDF2Image, NumPy, Matplotlib
