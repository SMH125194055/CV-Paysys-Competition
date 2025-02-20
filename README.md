
# Fingertip Detection & Fingerprint Extraction

## Overview

This project focuses on **Fingertip Detection** and **Fingerprint Extraction** using  **YOLO v11** . The challenge involved processing a dataset of right and left-hand palm images, training an object detection model, identifying fingertips, cropping them, and extracting biometric fingerprints.

## Approach

### 1. Model Selection & Training

* Leveraged experience with **YOLO models** from the Final Year Project.
* Used **YOLO v11** for precise fingertip detection.
* Utilized **Roboflow** for dataset management and preprocessing.
* Trained the model on a custom dataset for 30 epochs.

### 2. Image Processing & Feature Extraction

* Applied **OpenCV** for image preprocessing and segmentation.
* Cropped detected fingertip regions for further analysis.
* Extracted high-definition  **biometric fingerprint features** .

## Sample Images

### Cropped Fingertips

Here are some examples of the cropped fingertip images obtained after detection:

![Cropped Fingertip 1](https://chatgpt.com/c/Cropped_Fingerprints/cropped_fingertip1.png)
![Cropped Fingertip 2](https://chatgpt.com/c/Cropped_Fingerprints/cropped_fingertip2.png)
![Cropped Fingertip 3](https://chatgpt.com/c/Cropped_Fingerprints/cropped_fingertip3.png)
![Cropped Fingertip 4](https://chatgpt.com/c/Cropped_Fingerprints/cropped_fingertip4.png)
![Cropped Fingertip 5](https://chatgpt.com/c/Cropped_Fingerprints/cropped_fingertip5.png)

### Extracted Fingerprints

The following images showcase the extracted fingerprint features:

![Extracted Fingerprint 1](https://chatgpt.com/c/Extracted_Fingerprints/extracted_fingerprint1.png)
![Extracted Fingerprint 2](https://chatgpt.com/c/Extracted_Fingerprints/extracted_fingerprint2.png)
![Extracted Fingerprint 3](https://chatgpt.com/c/Extracted_Fingerprints/extracted_fingerprint3.png)
![Extracted Fingerprint 4](https://chatgpt.com/c/Extracted_Fingerprints/extracted_fingerprint4.png)
![Extracted Fingerprint 5](https://chatgpt.com/c/Extracted_Fingerprints/extracted_fingerprint5.png)

## Installation

To run this project, install the required dependencies:

```bash
pip install "ultralytics<=8.3.40" supervision roboflow opencv-python
```

## Dataset Preparation

Download the dataset using  **Roboflow** :

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("adpm").project("finger-tips-detection")
version = project.version(1)
dataset = version.download("yolov11")
```

## Training the Model

Train **YOLO v11** for fingertip detection:

```python
from ultralytics import YOLO
model = YOLO('yolo11s-seg.pt')

train_results = model.train(
    data='path/to/data.yaml',
    epochs=30,
    imgsz=640,
    device=0,
)
```

## Model Evaluation

Evaluate the trained model on validation data:

```python
metrics = model.val(data="path/to/data.yaml")
print(metrics)
```

## Results

* Successfully trained **YOLO v11** to detect fingertips with high accuracy.
* Extracted and enhanced **fingerprint images** using  **OpenCV** .
* Achieved robust performance in  **biometric feature extraction** .

## Future Improvements

* Experiment with **alternative YOLO architectures** for improved accuracy.
* Enhance fingerprint feature extraction using  **deep learning-based segmentation** .
* Integrate with **biometric authentication systems** for real-world applications.

## Contributors

* **[Your Name]** - Research & Implementation

## License

This project is open-source and available under the [MIT License](https://chatgpt.com/c/LICENSE).
