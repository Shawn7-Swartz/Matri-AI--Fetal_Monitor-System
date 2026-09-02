# 🩺 Matri-AI — AI-Powered Fetal Monitoring System

Matri-AI is an AI-based fetal monitoring and analysis system designed to assist in the analysis of fetal ultrasound images and videos. The system combines **deep learning, computer vision, and explainable AI** to provide automated fetal position analysis, fetal head detection, and visualization of model predictions.

The project provides an interactive **Streamlit-based interface** through which users can upload ultrasound images or videos and obtain AI-assisted analysis.

---

## ✨ Key Features

* 🧠 **Fetal Position Classification**

  * Deep-learning-based classification of fetal ultrasound images.
  * Uses **EfficientNetB0** and VGG16-based models.

* 🎯 **Fetal Head Detection**

  * YOLO-based object detection model for identifying the fetal head.
  * Achieved approximately **0.995 mAP@50** during evaluation.

* 🎥 **Image & Video Analysis**

  * Supports analysis through uploaded ultrasound images and videos.

* 🗂️ **DICOM Support**

  * Designed to support medical imaging data in DICOM format.

* 📈 **Contraction Estimation**

  * Uses pixel-level image variation across frames to estimate changes associated with uterine contractions.

* 🖥️ **Interactive Web Interface**

  * Built using **Streamlit** for simple interaction and visualization.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Ultrasound Input  │
                    │  Image / Video /    │
                    │       DICOM         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Preprocessing    │
                    │ Resize / Normalize  │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Fetal Position   │      │  Fetal Head      │
        │ Classification   │      │   Detection      │
        │ EfficientNet/VGG │      │      YOLO        │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                   ┌──────────────────────┐
                   │   Model Prediction   │
                   └──────────┬───────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             ┌──────────────┐    ┌──────────────┐
             │   Grad-CAM   │    │ Contraction  │
             │ Visualization│    │  Estimation  │
             └──────────────┘    └──────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │ Results & Analysis  │
                    └─────────────────────┘
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

Ultrasound images are preprocessed before being passed to the classification models.

The preprocessing pipeline includes:

* Image loading
* Resizing
* Normalization
* Data augmentation
* Dataset generation using `ImageDataGenerator`

---

### 2. Fetal Position Classification

The project experimented with deep convolutional neural network architectures including:

* **EfficientNetB0**
* **VGG16**

EfficientNetB0 was used as the primary architecture for the fetal position classification pipeline.

The trained model is stored as:

```text
fetal_position_model.h5
```

A test accuracy of approximately **87–89%** was achieved during model development and tuning.

---

### 3. Fetal Head Detection

A YOLO-based detector is used to locate the fetal head within ultrasound images.

The detection pipeline provides:

* Bounding-box localization
* Confidence scores
* Automated fetal head identification

The trained detector achieved approximately:

```text
mAP@50 ≈ 0.995
```

during evaluation.

---

## 🔬 Explainable AI — Grad-CAM

Medical AI systems should not only provide predictions but should also provide insight into **why a prediction was made**.

Matri-AI uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate heatmaps showing the regions of an ultrasound image that contributed most strongly to the model's prediction.

This allows users to visually inspect the model's attention rather than relying solely on the predicted class.

---

## 🎥 Video Analysis

Matri-AI can process ultrasound video input frame-by-frame.

The pipeline can:

1. Load the video.
2. Extract individual frames.
3. Preprocess frames.
4. Run the detection/classification models.
5. Analyze changes between frames.
6. Estimate contraction-related variation.
7. Present the resulting analysis through the Streamlit interface.

---

## 📊 Contraction Estimation

The system includes an experimental computer-vision-based approach for estimating contractions.

The approach analyzes **pixel-level variation between consecutive frames**.

Large changes between frames can be used as an indicator of movement or visual variation within the ultrasound sequence.

> This component is intended as an AI/computer-vision research feature and should not be interpreted as a clinical diagnostic measurement.

---

## 🛠️ Technology Stack

| Category             | Technologies       |
| -------------------- | ------------------ |
| Programming Language | Python             |
| Deep Learning        | TensorFlow / Keras |
| Computer Vision      | OpenCV             |
| ML                   | Scikit-learn       |
| Numerical Computing  | NumPy              |
| Data Processing      | Pandas             |
| Object Detection     | YOLO               |
| Medical Imaging      | DICOM              |
| Web Interface        | Streamlit          |
| Model Formats        | `.h5`              |
| Version Control      | Git / GitHub       |

---

## 📁 Project Structure

```text
Matri-AI--Fetal_Monitor-System/
│
├── Fetal_app.py
├── app.py
├── README.md
├── label_map.json
├── notes.exe
│
├── Models/
│   └── fetal_monitor_2.h5
│
└── assets/
    ├── Patient00781_Plane6_3_of_5.png
    ├── demo.keep
    ├── fetal.jpeg
    └── fetal.png
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Matri-AI--Fetal_Monitor-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages:

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install pandas
pip install scikit-learn
pip install streamlit
pip install pillow
```

---

## ▶️ Running the Application

Launch the Streamlit application using:

```bash
streamlit run app.py
```

or:

```bash
streamlit run Fetal_app.py
```

The application will provide an interactive interface for uploading ultrasound data and viewing the resulting AI analysis.

---

## 📈 Model Performance

| Component                     | Model          |           Performance |
| ----------------------------- | -------------- | --------------------: |
| Fetal Position Classification | EfficientNetB0 | ~87–89% test accuracy |
| Fetal Head Detection          | YOLO           |         ~0.995 mAP@50 |
| Classification                | EfficientNetB0 |      Test loss ~0.392 |

Performance values represent results obtained during project experimentation and may vary depending on the dataset split, preprocessing, and model configuration.

---

## 🧪 Dataset

The project was developed using fetal ultrasound imagery obtained from publicly available datasets, including datasets hosted through platforms such as **Kaggle and Zenodo**.

The dataset contains thousands of fetal ultrasound images representing different views/positions.

Dataset preprocessing and augmentation were performed before training the deep-learning models.

---

## 🔐 Medical Disclaimer

**Matri-AI is an academic/research project and is not a medical device.**

The predictions and visualizations generated by the system are intended for research and educational purposes only and should **not be used for medical diagnosis, treatment decisions, or clinical decision-making**.

A qualified medical professional should always interpret medical imaging and clinical findings.

---

## 🔮 Future Scope

Potential improvements include:

* Integration of larger and more diverse ultrasound datasets.
* Improved fetal position classification accuracy.
* Real-time ultrasound video analysis.
* More robust contraction detection.
* Multi-model ensemble approaches.
* Improved DICOM pipeline.
* Integration with hospital information systems.
* Cloud-based inference.
* Model optimization for edge devices.
* Additional explainability techniques.
* Clinical validation with expert annotations.

---

## 👩‍💻 Project

**Matri-AI — Fetal Monitoring System**

An academic AI/Computer Vision project exploring the use of deep learning and computer vision for automated fetal ultrasound analysis.

### Core Areas

`Artificial Intelligence` · `Machine Learning` · `Deep Learning` · `Computer Vision` · `Medical Imaging` · `Explainable AI` · `Object Detection`

---

## ⭐ Acknowledgements

This project builds upon publicly available fetal ultrasound datasets and open-source machine-learning and computer-vision technologies.

---

## 📜 License

This project is intended for academic and research purposes.


