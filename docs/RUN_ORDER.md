# Run Order

Recommended environment: Google Colab with GPU.

## 1. Install Requirements

In Colab, the notebook already installs `timm`. For a local/GPU environment, run:

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA, install the proper PyTorch build from the official PyTorch installation selector before running the full training workflow.

## 2. Prepare Dataset

The notebook expects the dataset ZIP here:

```python
zip_file_path = '/content/gdrive/MyDrive/archive (2).zip'
```

Update this path if your dataset has a different filename or location.

## 3. Execute Notebook

Run:

```text
notebooks/ensemble_base_accuracy_clean.ipynb
```

Suggested execution order:

1. Install/import packages
2. Mount Google Drive
3. Extract dataset
4. Load image paths and labels
5. Remove duplicate images using file hashes
6. Create clean stratified train/test split
7. Prepare 5-fold cross-validation
8. Train individual pretrained models
9. Evaluate fold models
10. Evaluate ensemble models
11. Run feature fusion models
12. Generate final comparison table

## 4. Output Directory

The notebook writes outputs to:

```python
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
```

Large files such as `.pt`, `.pth`, `.h5`, checkpoints, and full output folders are intentionally ignored by Git.
