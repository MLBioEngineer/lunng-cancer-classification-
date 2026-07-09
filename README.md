# Lung Cancer CT Image Classification using Transfer Learning and Ensemble Learning

This repository contains a deep learning pipeline for lung cancer CT image classification using pretrained CNN architectures, 5-fold cross-validation, ensemble prediction, feature fusion, Grad-CAM visualization, and uncertainty analysis.

## Project Overview

The project classifies lung CT images into four categories:

- Adenocarcinoma
- Large cell carcinoma
- Normal
- Squamous cell carcinoma

The uploaded notebook was cleaned for GitHub by removing heavy cell outputs while preserving the full code workflow.

## Key Methods

- Duplicate image leakage checking using MD5 file hashes
- Clean train/validation/test split after duplicate removal
- Stratified 5-fold cross-validation
- Transfer learning using `timm` pretrained models
- CNN ensemble evaluation on the final clean test set
- Feature-fusion models
- Grad-CAM visualization
- ROC-AUC and uncertainty analysis

## Best Result

The best final clean test-set performance was achieved by **DenseNet201 5-fold ensemble**:

| Metric | Score |
|---|---:|
| Accuracy | 0.9941 |
| Precision | 0.9943 |
| Recall | 0.9941 |
| F1 Score | 0.9941 |

## Final Model Comparison

| Rank | Model | Accuracy | Precision | Recall | F1 Score |
|---:|---|---:|---:|---:|---:|
| 1 | DenseNet201 Ensemble | 0.9941 | 0.9943 | 0.9941 | 0.9941 |
| 2 | DenseNet201 No-Aug Ensemble | 0.9882 | 0.9886 | 0.9882 | 0.9882 |
| 3 | DenseNet121 No-Aug Ensemble | 0.9765 | 0.9770 | 0.9765 | 0.9762 |
| 4 | MobileNetV2 Ensemble | 0.9765 | 0.9765 | 0.9765 | 0.9765 |
| 5 | MobileNetV3 Large No-Aug Ensemble | 0.9765 | 0.9768 | 0.9765 | 0.9764 |

Full result tables are available in [`results/final_model_comparison.csv`](results/final_model_comparison.csv) and [`results/cv_summary.csv`](results/cv_summary.csv).

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── ensemble_base_accuracy_clean.ipynb
├── scripts/
│   └── ensemble_base_accuracy_extracted.py
├── docs/
│   ├── DATASET.md
│   ├── RESULTS.md
│   ├── RUN_ORDER.md
│   └── GITHUB_UPLOAD_CHECKLIST.md
└── results/
    ├── final_model_comparison.csv
    └── cv_summary.csv
```

## How to Run

Google Colab with GPU is recommended.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the dataset ZIP in Google Drive or update this path inside the notebook:

```python
zip_file_path = '/content/gdrive/MyDrive/archive (2).zip'
```

3. Run the notebook:

```text
notebooks/ensemble_base_accuracy_clean.ipynb
```

4. Results will be generated in the configured output directory:

```python
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
```

## Notes

- Dataset files are not included in this repository.
- Trained model weights/checkpoints are not included because they are usually large.
- Notebook outputs were removed to keep the repository lightweight and upload-friendly.
- This project is for research and educational use only; it is not a clinical diagnostic system.
