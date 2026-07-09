# Dataset Guide

Dataset files are not included in this repository.

## Expected Original Structure

The notebook expects a ZIP file that extracts into:

```text
/content/unzipped_data/Data/
├── train/
│   ├── normal/
│   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
│   ├── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
│   └── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
├── valid/
│   └── same class folders
└── test/
    └── same class folders
```

The notebook maps long folder names into four clean labels:

```text
normal
adenocarcinoma
squamous.cell.carcinoma
large.cell.carcinoma
```

## Leakage Control

The notebook combines the original train/valid/test files, removes duplicate images using MD5 hashes, and creates a new clean stratified split.

Reported from the uploaded notebook:

- Original images: 1000
- Clean images after duplicate removal: 847
- Removed duplicates: 153
