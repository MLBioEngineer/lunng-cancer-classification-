# Results Summary

The results below were extracted from the uploaded notebook output before cleaning it for GitHub.

## Best Final Test Result

**DenseNet201 5-fold ensemble** achieved the best clean final test-set performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.994118 |
| Precision | 0.994272 |
| Recall | 0.994118 |
| F1 Score | 0.994135 |

## Final Clean Test-Set Comparison

See the full CSV file here: `results/final_model_comparison.csv`.

Top models:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| densenet201_lung_clean (Ensemble) | 0.994118 | 0.994272 | 0.994118 | 0.994135 |
| densenet201_lung_clean_no_aug (Ensemble) | 0.988235 | 0.988571 | 0.988235 | 0.988163 |
| densenet121_lung_clean_no_aug (Ensemble) | 0.976471 | 0.976975 | 0.976471 | 0.976151 |
| mobilenetv2_lung_clean (Ensemble) | 0.976471 | 0.976471 | 0.976471 | 0.976471 |
| mobilenetv3_large_100_lung_clean_no_aug (Ensemble) | 0.976471 | 0.976811 | 0.976471 | 0.976403 |

## 5-Fold Cross-Validation Summary

See the full CSV file here: `results/cv_summary.csv`.

| Model | CV Mean Accuracy | CV Std Accuracy |
|---|---:|---:|
| fusion_mobilenetv2_densenet201_clean | 0.983791 | 0.011754 |
| fusion_mobilenetv3_densenet121_clean | 0.982298 | 0.016571 |
| densenet201_lung_clean | 0.982266 | 0.003665 |
| mobilenetv3_large_100_lung_clean | 0.964553 | 0.022648 |
| efficientnetv2_s_lung_clean | 0.961601 | 0.005490 |
| densenet121_lung_clean | 0.960153 | 0.011872 |
| mobilenetv2_lung_clean | 0.958671 | 0.015139 |

## Data Cleaning Summary

- Original images: 1000
- Images after duplicate removal: 847
- Removed duplicates: 153
- Final test set: 20% stratified split after duplicate removal
- Number of classes: 4

Class labels:

- adenocarcinoma
- large.cell.carcinoma
- normal
- squamous.cell.carcinoma
