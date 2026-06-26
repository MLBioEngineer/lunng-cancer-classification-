# Lung Cancer Classification Using Deep Learning

This repository contains a deep learning pipeline for lung cancer image classification using multiple CNN architectures and feature fusion.

## Project Overview

The project includes:

- Dataset extraction and preprocessing
- Duplicate image removal using file hashing
- Clean train/validation/test splitting
- 5-fold cross-validation
- Individual CNN model training
- Ensemble evaluation
- Feature fusion models
- ROC/AUC analysis
- Confusion matrix visualization
- Grad-CAM explainability
- Uncertainty analysis

## Models Used

- MobileNetV2
- MobileNetV3 Large
- DenseNet121
- DenseNet201
- EfficientNetV2-S
- MobileNetV3 + DenseNet121 feature fusion
- MobileNetV2 + DenseNet201 feature fusion

## Repository Structure

```text
lung-cancer-classification/
│
├── lung_cancer_classification.py   # Main project code converted from Colab notebook
├── requirements.txt                # Required Python packages
├── .gitignore                      # Ignored files and folders
└── README.md                       # Project documentation
```

## Dataset

The dataset is not included in this repository because medical image datasets and trained model files are usually large.

Update the dataset path inside the Python file before running:

```python
zip_file_path = "path/to/archive.zip"
```

The original code was developed in Google Colab and uses Google Drive paths.

## How to Run

### Option 1: Google Colab

1. Upload `lung_cancer_classification.py` or copy the code into a Colab notebook.
2. Mount Google Drive.
3. Set your dataset path.
4. Run the cells/sections sequentially.

### Option 2: Local Python Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python lung_cancer_classification.py
```

Note: The script was originally exported from a Colab notebook, so you may need to adjust Google Drive paths and Colab-specific commands before running locally.


