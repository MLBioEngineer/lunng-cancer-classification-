"""
Lung cancer CT image classification using transfer learning and ensemble learning.

Auto-extracted from notebooks/ensemble_base_accuracy_clean.ipynb.
Primary execution environment: Google Colab or a GPU-enabled Python environment.
Dataset and trained model files are intentionally not included in this repository.
"""

# %% Cell 1
# Notebook command: !pip install -q timm

# %% Cell 2
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# %% Cell 3
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import os
import json
import copy
import random
import zipfile
import hashlib
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
set_all_seeds(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% Cell 4
zip_file_path = '/content/gdrive/MyDrive/archive (2).zip'
extract_dir = '/content/unzipped_data'

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Dataset extracted to: {extract_dir}")

# %% Cell 5
base_data_dir = '/content/unzipped_data/Data'
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

def find_images_and_labels_from_subdirs(root_dir, extensions):
    image_paths = []
    labels = []

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)

        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(extensions):
                    image_paths.append(os.path.join(class_dir, file))
                    labels.append(class_name)

    return image_paths, labels

train_paths, train_labels = find_images_and_labels_from_subdirs(
    os.path.join(base_data_dir, 'train'),
    image_extensions
)

valid_paths, valid_labels = find_images_and_labels_from_subdirs(
    os.path.join(base_data_dir, 'valid'),
    image_extensions
)

test_paths, test_labels = find_images_and_labels_from_subdirs(
    os.path.join(base_data_dir, 'test'),
    image_extensions
)

train_initial_df = pd.DataFrame({
    'filepath': train_paths,
    'label': train_labels,
    'original_split': 'train'
})

valid_initial_df = pd.DataFrame({
    'filepath': valid_paths,
    'label': valid_labels,
    'original_split': 'valid'
})

test_initial_df = pd.DataFrame({
    'filepath': test_paths,
    'label': test_labels,
    'original_split': 'test'
})

print("Original train:", len(train_initial_df))
print("Original valid:", len(valid_initial_df))
print("Original test :", len(test_initial_df))

# %% Cell 6
label_mapping = {
    'normal':                                            'normal',
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib':       'adenocarcinoma',
    'adenocarcinoma':                                    'adenocarcinoma',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'squamous.cell.carcinoma',
    'squamous.cell.carcinoma':                          'squamous.cell.carcinoma',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa':    'large.cell.carcinoma',
    'large.cell.carcinoma':                             'large.cell.carcinoma',
}

all_df = pd.concat(
    [train_initial_df, valid_initial_df, test_initial_df],
    ignore_index=True
)

all_df['label'] = all_df['label'].map(label_mapping)
all_df = all_df.dropna(subset=['label']).reset_index(drop=True)

print("Total images before duplicate removal:", len(all_df))
print("\nClass distribution before duplicate removal:")
print(all_df['label'].value_counts())

# %% Cell 7
def file_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()

all_df['file_hash'] = all_df['filepath'].apply(file_hash)

print("Total rows:", len(all_df))
print("Unique hashes:", all_df['file_hash'].nunique())
print("Exact duplicate rows:", len(all_df) - all_df['file_hash'].nunique())

# %% Cell 8
conflict_hashes = (
    all_df.groupby('file_hash')['label']
    .nunique()
    .reset_index()
)

conflict_hashes = conflict_hashes[conflict_hashes['label'] > 1]

print("Conflicting duplicate hashes:", len(conflict_hashes))

if len(conflict_hashes) > 0:
    conflict_examples = all_df[all_df['file_hash'].isin(conflict_hashes['file_hash'])]
    display(conflict_examples.sort_values('file_hash').head(50))

# %% Cell 9
clean_df = all_df.drop_duplicates(subset=['file_hash']).reset_index(drop=True)

print("Total images after duplicate removal:", len(clean_df))
print("Removed duplicates:", len(all_df) - len(clean_df))

print("\nClean class distribution:")
print(clean_df['label'].value_counts())

# %% [markdown] Cell 10
# split

# %% Cell 11
clean_train_val_df, clean_test_df = train_test_split(
    clean_df,
    test_size=0.20,
    stratify=clean_df['label'],
    random_state=SEED
)

clean_train_val_df = clean_train_val_df.reset_index(drop=True)
clean_test_df = clean_test_df.reset_index(drop=True)

print("Clean Train+Val:", len(clean_train_val_df))
print("Clean Final Test:", len(clean_test_df))

print("\nClean Train+Val distribution:")
print(clean_train_val_df['label'].value_counts())

print("\nClean Final Test distribution:")
print(clean_test_df['label'].value_counts())

# %% [markdown] Cell 12
# duplivcate hasshes

# %% Cell 13
def check_hash_overlap(df1, df2, name1, name2):
    overlap = set(df1['file_hash']) & set(df2['file_hash'])
    print(f"{name1} vs {name2}: {len(overlap)} duplicate hashes")

check_hash_overlap(clean_train_val_df, clean_test_df, "Clean TrainVal", "Clean Final Test")

# %% Cell 14
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

clean_train_val_csv = os.path.join(gdrive_base_output_dir, "clean_train_val_split.csv")
clean_test_csv = os.path.join(gdrive_base_output_dir, "clean_final_test_split.csv")

clean_train_val_df.to_csv(clean_train_val_csv, index=False)
clean_test_df.to_csv(clean_test_csv, index=False)

print("Saved:", clean_train_val_csv)
print("Saved:", clean_test_csv)

# %% [markdown] Cell 15
# class names

# %% Cell 16
class_names = sorted(clean_df['label'].unique().tolist())
num_classes = len(class_names)
labels_map = {label: i for i, label in enumerate(class_names)}

print("Class names:", class_names)
print("Num classes:", num_classes)
print("Labels map:", labels_map)

# %% [markdown] Cell 17
# configure set up

# %% Cell 18
COMMON_CONFIG = {
    # general
    'input_ch': 3,
    'ImageNet': True,
    'num_folds': 5,
    'Resize_h': 224,
    'Resize_w': 224,
    'stop_criteria': 'loss',

    # same training hyperparameters for all individual models
    'lr': 0.0001,
    'optim_fc': 'Adam',
    'batch_size': 16,
    'n_epochs': 100,
    'max_epochs_stop': 30,
    'epochs_patience': 10,

    # class info
    'num_classes': num_classes,
    'class_names': class_names,
    'labels_map': labels_map,

    # augmentation
    'aug_random_resized_crop_scale_min': 0.88,
    'aug_random_resized_crop_scale_max': 1.00,
    'aug_random_rotation_degrees': 7,
    'aug_color_jitter_brightness': 0.08,
    'aug_color_jitter_contrast': 0.10,
    'aug_color_jitter_saturation': 0.0,
    'aug_color_jitter_hue': 0.0,
    'aug_random_affine_translate_max': 0.04,
    'aug_random_affine_scale_min': 0.96,
    'aug_random_affine_scale_max': 1.04,
    'aug_random_affine_shear': 3,
    'aug_random_vertical_flip_p': 0.0,
    'aug_random_grayscale_p': 0.0,
    'aug_gaussian_blur_p': 0.08,
    'aug_random_perspective_p': 0.0,
    'aug_random_erasing_p': 0.12,

    # regularization
    'label_smoothing': 0.05,

    # individual CNN model setting
    'freeze_backbone': False,
    'num_unfrozen_layers_backbone': 0,
}

print("COMMON_CONFIG ready.")

# %% [markdown] Cell 19
# augmented code

# %% Cell 20
def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def build_train_transform(config):
    return transforms.Compose([
        transforms.Resize((256, 256)),

        transforms.RandomResizedCrop(
            size=(config['Resize_h'], config['Resize_w']),
            scale=(
                config['aug_random_resized_crop_scale_min'],
                config['aug_random_resized_crop_scale_max']
            ),
            ratio=(0.95, 1.05)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=config['aug_random_rotation_degrees']
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(
                config['aug_random_affine_translate_max'],
                config['aug_random_affine_translate_max']
            ),
            scale=(
                config['aug_random_affine_scale_min'],
                config['aug_random_affine_scale_max']
            ),
            shear=config['aug_random_affine_shear']
        ),

        transforms.ColorJitter(
            brightness=config['aug_color_jitter_brightness'],
            contrast=config['aug_color_jitter_contrast'],
            saturation=0.0,
            hue=0.0
        ),

        transforms.RandomAutocontrast(p=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.15),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=config['aug_gaussian_blur_p']),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=as_list(config['mean_overall']),
            std=as_list(config['std_overall'])
        ),

        transforms.RandomErasing(
            p=config['aug_random_erasing_p'],
            scale=(0.01, 0.06),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

# %% Cell 21
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, labels_map=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels_map = labels_map

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')

        label_name = self.dataframe.iloc[idx]['label']
        label = self.labels_map[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# %% Cell 22
def compute_mean_std_from_dataframe(dataframe):
    means, stds = [], []

    for img_path in tqdm(dataframe['filepath'], desc="Computing mean/std"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img) / 255.0

            means.append(img_np.mean(axis=(0, 1)))
            stds.append(img_np.std(axis=(0, 1)))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    mean_overall = np.mean(means, axis=0)
    std_overall = np.mean(stds, axis=0)

    print("Mean:", mean_overall)
    print("Std :", std_overall)

    return mean_overall, std_overall

# %% Cell 23
def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

def build_train_transform(config):
    return transforms.Compose([
        transforms.Resize((256, 256)),

        transforms.RandomResizedCrop(
            size=(config['Resize_h'], config['Resize_w']),
            scale=(
                config['aug_random_resized_crop_scale_min'],
                config['aug_random_resized_crop_scale_max']
            ),
            ratio=(0.95, 1.05)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=config['aug_random_rotation_degrees']
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(
                config['aug_random_affine_translate_max'],
                config['aug_random_affine_translate_max']
            ),
            scale=(
                config['aug_random_affine_scale_min'],
                config['aug_random_affine_scale_max']
            ),
            shear=config['aug_random_affine_shear']
        ),

        transforms.ColorJitter(
            brightness=config['aug_color_jitter_brightness'],
            contrast=config['aug_color_jitter_contrast'],
            saturation=0.0,
            hue=0.0
        ),

        transforms.RandomAutocontrast(p=0.15),
        transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.15),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=config['aug_gaussian_blur_p']),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=as_list(config['mean_overall']),
            std=as_list(config['std_overall'])
        ),

        transforms.RandomErasing(
            p=config['aug_random_erasing_p'],
            scale=(0.01, 0.06),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

def build_val_test_transform(config):
    return transforms.Compose([
        transforms.Resize((config['Resize_h'], config['Resize_w'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=as_list(config['mean_overall']),
            std=as_list(config['std_overall'])
        )
    ])

print("Dataset and transforms ready.")

# %% Cell 24
def make_json_serializable_config(config):
    cfg = {}

    for k, v in config.items():
        if isinstance(v, np.ndarray):
            cfg[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            cfg[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            cfg[k] = int(v)
        else:
            cfg[k] = v

    return cfg
def prepare_clean_folds(clean_train_val_df, base_config, output_dir):
    skf = StratifiedKFold(
        n_splits=base_config['num_folds'],
        shuffle=True,
        random_state=SEED
    )

    folds_info = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(clean_train_val_df['filepath'], clean_train_val_df['label'])
    ):
        fold_num = fold + 1

        print(f"\nPreparing Fold {fold_num}/{base_config['num_folds']}")

        train_fold_df = clean_train_val_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = clean_train_val_df.iloc[val_idx].reset_index(drop=True)

        hash_overlap = set(train_fold_df['file_hash']) & set(val_fold_df['file_hash'])
        print("Train-Val duplicate hash overlap:", len(hash_overlap))

        if len(hash_overlap) > 0:
            raise ValueError(f"Leakage found inside Fold {fold_num}")

        # Strict normalization: train fold only
        mean_fold, std_fold = compute_mean_std_from_dataframe(train_fold_df)

        fold_config = base_config.copy()
        fold_config['mean_overall'] = mean_fold
        fold_config['std_overall'] = std_fold

        # save fold split
        split_dir = os.path.join(output_dir, "fold_splits")
        os.makedirs(split_dir, exist_ok=True)

        train_fold_df.to_csv(
            os.path.join(split_dir, f"fold_{fold_num}_train.csv"),
            index=False
        )

        val_fold_df.to_csv(
            os.path.join(split_dir, f"fold_{fold_num}_val.csv"),
            index=False
        )

        # save fold config
        fold_config_json = make_json_serializable_config(fold_config)

        with open(os.path.join(split_dir, f"fold_{fold_num}_config.json"), "w") as f:
            json.dump(fold_config_json, f, indent=4)

        folds_info.append({
            'fold': fold_num,
            'train_df': train_fold_df,
            'val_df': val_fold_df,
            'config': fold_config
        })

    return folds_info
folds_info = prepare_clean_folds(
    clean_train_val_df=clean_train_val_df,
    base_config=COMMON_CONFIG,
    output_dir=gdrive_base_output_dir
)

print("All clean folds prepared.")

# %% Cell 25
print(f"Number of folds: {COMMON_CONFIG['num_folds']}")

for i, fold_data in enumerate(folds_info):
    fold_num = fold_data['fold']
    train_df = fold_data['train_df']
    val_df = fold_data['val_df']

    print(f"\nFold {fold_num}:")
    print(f"  Number of training images: {len(train_df)}")
    print(f"  Number of validation images: {len(val_df)}")
    print(f"  Training class distribution:\n{train_df['label'].value_counts()}")
    print(f"  Validation class distribution:\n{val_df['label'].value_counts()}")

# %% [markdown] Cell 26
# Training, evaluation, and plotting functions

# %% Cell 27
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, save_path=None):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    for epoch in range(config['n_epochs']):
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']} Train"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']} Val"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        print(
            f"Epoch {epoch+1}/{config['n_epochs']} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved: {save_path}")

        else:
            epochs_no_improve += 1
            print(f"No improvement: {epochs_no_improve}/{config['epochs_patience']}")

            if epochs_no_improve >= config['epochs_patience']:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model, history
def evaluate_model(model, dataloader, device, class_names, title="Evaluation", show_plots=True):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=title):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n--- {title} Metrics ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    if show_plots:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f"{title} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], '--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title} ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    if num_folds == 1:
        axes = [axes]

    fig.suptitle(f"Training and Validation Curves: {model_name}", fontsize=16)

    for i, history in enumerate(histories):
        axes[i][0].plot(history['train_loss'], label='Train Loss')
        axes[i][0].plot(history['val_loss'], label='Val Loss')
        axes[i][0].set_title(f"Fold {i+1} Loss")
        axes[i][0].set_xlabel("Epoch")
        axes[i][0].set_ylabel("Loss")
        axes[i][0].legend()
        axes[i][0].grid(True)

        axes[i][1].plot(history['train_acc'], label='Train Accuracy')
        axes[i][1].plot(history['val_acc'], label='Val Accuracy')
        axes[i][1].set_title(f"Fold {i+1} Accuracy")
        axes[i][1].set_xlabel("Epoch")
        axes[i][1].set_ylabel("Accuracy")
        axes[i][1].legend()
        axes[i][1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %% [markdown] Cell 28
# : Build and train individual models

# %% Cell 29
def build_timm_model(model_to_load, config):
    model = timm.create_model(
        model_to_load,
        pretrained=config['ImageNet'],
        num_classes=config['num_classes']
    )

    if config['freeze_backbone']:
        print(f"Freezing backbone for {model_to_load}")
        for name, param in model.named_parameters():
            if ('classifier' not in name) and ('head' not in name) and ('fc' not in name):
                param.requires_grad = False
    else:
        print(f"No freezing for {model_to_load}. All layers trainable.")
        for param in model.parameters():
            param.requires_grad = True

    return model
def train_single_model_same_config(model_spec, folds_info, base_output_dir):
    model_to_load = model_spec['model_to_load']
    model_name = model_spec['model_name']

    model_save_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    histories = []

    print(f"\n==============================")
    print(f"Training model: {model_name}")
    print(f"Backbone: {model_to_load}")
    print(f"Same config for all individual models")
    print(f"==============================")

    for fold_info in folds_info:
        fold = fold_info['fold']
        train_fold_df = fold_info['train_df']
        val_fold_df = fold_info['val_df']

        fold_config = fold_info['config'].copy()
        fold_config['model_to_load'] = model_to_load
        fold_config['model_name'] = model_name
        fold_config['save_path'] = model_save_dir

        print(f"\nStarting Fold {fold}/{fold_config['num_folds']} for {model_name}")

        hash_overlap = set(train_fold_df['file_hash']) & set(val_fold_df['file_hash'])
        print("Train-Val duplicate hash overlap:", len(hash_overlap))

        if len(hash_overlap) > 0:
            raise ValueError("Leakage found inside fold. Stop training.")

        transform_train = build_train_transform(fold_config)
        transform_val = build_val_test_transform(fold_config)

        train_dataset = CustomImageDataset(
            train_fold_df,
            transform=transform_train,
            labels_map=fold_config['labels_map']
        )

        val_dataset = CustomImageDataset(
            val_fold_df,
            transform=transform_val,
            labels_map=fold_config['labels_map']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        model = build_timm_model(model_to_load, fold_config)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(
            label_smoothing=fold_config.get('label_smoothing', 0.0)
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=fold_config['lr']
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=fold_config['epochs_patience'] // 2
        )

        fold_save_path = os.path.join(
            model_save_dir,
            f"{model_name}_fold_{fold}.pt"
        )

        model_trained, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=fold_config,
            device=device,
            save_path=fold_save_path
        )

        histories.append(history)

        print(f"\nValidation evaluation for {model_name} Fold {fold}")
        evaluate_model(
            model_trained,
            val_loader,
            device,
            fold_config['class_names'],
            title=f"{model_name} Fold {fold} Validation",
            show_plots=True
        )

        # save fold config
        fold_config_to_save = make_json_serializable_config(fold_config)
        fold_config_save_path = os.path.join(
            model_save_dir,
            f"{model_name}_fold_{fold}_config.json"
        )

        with open(fold_config_save_path, "w") as f:
            json.dump(fold_config_to_save, f, indent=4)

    history_save_path = os.path.join(
        base_output_dir,
        f"{model_name}_histories_clean.json"
    )

    with open(history_save_path, "w") as f:
        json.dump(histories, f, indent=4)

    print(f"\nSaved history: {history_save_path}")

    plot_training_curves(histories, model_name)

    return histories

# %% [markdown] Cell 30
# 5 fold CV MEAN+-STD deaviation

# %% Cell 31
import os
import json
import numpy as np
import glob

# Ensure the correct output directory is used
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

print("=" * 75)
print("5-Fold Cross-Validation Report (Best Validation Accuracy)")
print("=" * 75)
print(f"{'Model Name':<50} | {'CV Accuracy (Mean ± Std)'}")
print("-" * 75)

# Find all saved history JSON files
history_files = glob.glob(os.path.join(gdrive_base_output_dir, '*histories*.json'))

results = []

for file_path in sorted(history_files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle both list and dict formats for histories based on how they were saved
        models_to_process = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0 and 'val_acc' in v[0]:
                    models_to_process[k] = v
        elif isinstance(data, list) and len(data) > 0 and 'val_acc' in data[0]:
            # Extract model name from filename
            m_name = os.path.basename(file_path).replace('_histories_clean.json', '').replace('_all_folds_histories.json', '')
            models_to_process[m_name] = data

        for m_name, histories in models_to_process.items():
            # Get the max validation accuracy for each fold
            best_accs = [np.max(fold['val_acc']) for fold in histories if 'val_acc' in fold]
            if best_accs:
                mean_acc = np.mean(best_accs)
                std_acc = np.std(best_accs)
                results.append((m_name, mean_acc, std_acc))

    except Exception as e:
        continue

# Sort results by mean accuracy in descending order
results.sort(key=lambda x: x[1], reverse=True)

for m_name, mean_acc, std_acc in results:
    if m_name == 'fusion_effnetv2_convnext_clean':
        continue
    print(f"{m_name:<50} | {mean_acc:.4f} ± {std_acc:.4f}")
print("=" * 75)

# %% Cell 32
import pandas as pd
import os

# Filter the results to remove the excluded model
filtered_results = [
    {"Model Name": m_name, "CV Mean Accuracy": mean_acc, "CV Std Accuracy": std_acc}
    for m_name, mean_acc, std_acc in results
    if m_name != 'fusion_effnetv2_convnext_clean'
]

# Create a DataFrame
df_results = pd.DataFrame(filtered_results)

# Define the save path
save_path = os.path.join(gdrive_base_output_dir, '5_fold_cv_summary_report.csv')

# Save to CSV
df_results.to_csv(save_path, index=False)

print(f"Saved CV report to: {save_path}")
display(df_results)

# %% [markdown] Cell 33
# mobilenetv2_100"

# %% Cell 34
import os

# Define the model specifications for mobilenetv2_100
MODEL_SPECS_MOBILENETV2 = [
    {
        'model_to_load': "mobilenetv2_100",
        'model_name': "mobilenetv2_lung_clean",
    }
]

# Ensure gdrive_base_output_dir exists (already defined in previous cells)
os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

# Dictionary to store training histories for all models
all_model_histories_mobilenetv2 = {}

# Loop through each model specification and train the model using K-fold cross-validation
for model_spec in MODEL_SPECS_MOBILENETV2:
    model_name = model_spec['model_name']
    print(f"\n{'='*50}")
    print(f"Initiating K-fold training for {model_name}")
    print(f"{'-'*50}")

    # Call the train_single_model_same_config function
    # folds_info is available from previous cells and contains the common config for all folds
    histories = train_single_model_same_config(
        model_spec=model_spec,
        folds_info=folds_info,
        base_output_dir=gdrive_base_output_dir
    )

    all_model_histories_mobilenetv2[model_name] = histories

print("\n" + "="*50)
print("MobileNetV2 model processed. 'all_model_histories_mobilenetv2' dictionary contains training histories.")
print(f"Keys in all_model_histories_mobilenetv2: {list(all_model_histories_mobilenetv2.keys())}")
print("="*50)

# %% [markdown] Cell 35
# ### Evaluate All MobileNetV2 Folds on Test Set

# %% Cell 36
import torch
import os
import json
import numpy as np
import timm

set_all_seeds(SEED)

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Model details for MobileNetV2
model_to_save_name_mobilenetv2 = 'mobilenetv2_lung_clean'
model_to_load_architecture_mobilenetv2 = 'mobilenetv2_100'

all_test_results_mobilenetv2 = []

print(f"\nEvaluating all {COMMON_CONFIG['num_folds']} MobileNetV2 fold models on the final test set...")

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    print(f"\n--- Evaluating MobileNetV2 Fold {fold_number} on the test set ---")
    model_save_dir_mobilenetv2 = os.path.join(gdrive_base_output_dir, model_to_save_name_mobilenetv2)
    model_path_to_load_mobilenetv2 = os.path.join(
        model_save_dir_mobilenetv2,
        f"{model_to_save_name_mobilenetv2}_fold_{fold_number}.pt"
    )
    fold_config_path_mobilenetv2 = os.path.join(
        model_save_dir_mobilenetv2,
        f"{model_to_save_name_mobilenetv2}_fold_{fold_number}_config.json"
    )

    try:
        # Load fold-specific configuration
        with open(fold_config_path_mobilenetv2, 'r') as f:
            fold_config_mobilenetv2 = json.load(f)

        # Convert mean/std back to numpy arrays
        fold_config_mobilenetv2['mean_overall'] = np.array(fold_config_mobilenetv2['mean_overall'])
        fold_config_mobilenetv2['std_overall'] = np.array(fold_config_mobilenetv2['std_overall'])

        # Build the model using the fold's config directly with timm
        model_mobilenetv2 = timm.create_model(
            model_to_load_architecture_mobilenetv2,
            pretrained=False,
            num_classes=fold_config_mobilenetv2.get('num_classes', 4)
        )
        model_mobilenetv2 = model_mobilenetv2.to(device)

        # Load the state dictionary
        model_mobilenetv2.load_state_dict(torch.load(model_path_to_load_mobilenetv2, map_location=device))
        model_mobilenetv2.eval() # Ensure the model is in evaluation mode

        # Build test transformations using the fold-specific mean/std
        transform_test_mobilenetv2 = build_val_test_transform(fold_config_mobilenetv2)

        # Create CustomImageDataset and DataLoader for the test set
        test_dataset_mobilenetv2 = CustomImageDataset(
            clean_test_df,
            transform=transform_test_mobilenetv2,
            labels_map=fold_config_mobilenetv2['labels_map']
        )

        test_loader_mobilenetv2 = DataLoader(
            test_dataset_mobilenetv2,
            batch_size=fold_config_mobilenetv2['batch_size'],
            shuffle=False,
            num_workers=2
        )

        # Evaluate the model on the test set
        test_results_fold = evaluate_model(
            model=model_mobilenetv2,
            dataloader=test_loader_mobilenetv2,
            device=device,
            class_names=fold_config_mobilenetv2['class_names'],
            title=f"MobileNetV2 Fold {fold_number} Final Test Set Evaluation",
            show_plots=False # Only show plots for overall evaluation if needed, not for each fold
        )
        all_test_results_mobilenetv2.append(test_results_fold['accuracy'])

    except FileNotFoundError:
        print(f"Error: Model or config file not found for Fold {fold_number} at {model_path_to_load_mobilenetv2} or {fold_config_path_mobilenetv2}")
    except Exception as e:
        print(f"An error occurred while loading or evaluating MobileNetV2 Fold {fold_number}: {e}")

if all_test_results_mobilenetv2:
    cumulative_accuracy_mobilenetv2 = np.mean(all_test_results_mobilenetv2)
    print(f"\nCumulative Average Accuracy of MobileNetV2 across all folds on the test set: {cumulative_accuracy_mobilenetv2:.4f}")

    # Optionally, save the cumulative results
    output_file_path_mobilenetv2 = os.path.join(
        gdrive_base_output_dir,
        f'{model_to_save_name_mobilenetv2}_cumulative_test_results.json'
    )
    with open(output_file_path_mobilenetv2, 'w') as f:
        json.dump({
            'average_accuracy': cumulative_accuracy_mobilenetv2,
            'fold_accuracies': [float(acc) for acc in all_test_results_mobilenetv2]
        }, f, indent=4)
    print(f"Cumulative MobileNetV2 test results saved to: {output_file_path_mobilenetv2}")
else:
    print("No MobileNetV2 test results were collected.")

# %% Cell 37
print(f"Cumulative Average Accuracy of MobileNetV2 across all folds on the test set: {cumulative_accuracy_mobilenetv2:.4f}")

# %% Cell 38
import torch
import os
import json
import numpy as np
import timm

# Define the model details for MobileNetV2
model_to_save_name_mobilenetv2 = 'mobilenetv2_lung_clean'
model_to_load_architecture_mobilenetv2 = 'mobilenetv2_100'

# Assuming we want to save the model from the last trained fold (e.g., fold 5)
fold_number_to_save = 5

# Reconstruct the path where the MobileNetV2 model for the specific fold was saved during training
model_save_dir_mobilenetv2 = os.path.join(gdrive_base_output_dir, model_to_save_name_mobilenetv2)
model_path_to_load_mobilenetv2 = os.path.join(
    model_save_dir_mobilenetv2,
    f"{model_to_save_name_mobilenetv2}_fold_{fold_number_to_save}.pt"
)
fold_config_path_mobilenetv2 = os.path.join(
    model_save_dir_mobilenetv2,
    f"{model_to_save_name_mobilenetv2}_fold_{fold_number_to_save}_config.json"
)

# Load the fold-specific configuration for building the model
try:
    with open(fold_config_path_mobilenetv2, 'r') as f:
        fold_config_mobilenetv2 = json.load(f)

    # Convert mean/std back to numpy arrays (json stores them as lists)
    if 'mean_overall' in fold_config_mobilenetv2 and isinstance(fold_config_mobilenetv2['mean_overall'], list):
        fold_config_mobilenetv2['mean_overall'] = np.array(fold_config_mobilenetv2['mean_overall'])
    if 'std_overall' in fold_config_mobilenetv2 and isinstance(fold_config_mobilenetv2['std_overall'], list):
        fold_config_mobilenetv2['std_overall'] = np.array(fold_config_mobilenetv2['std_overall'])

    # Build the MobileNetV2 model architecture directly using timm
    final_model_mobilenetv2 = timm.create_model(
        model_to_load_architecture_mobilenetv2,
        pretrained=False,
        num_classes=fold_config_mobilenetv2.get('num_classes', 4)
    )
    final_model_mobilenetv2 = final_model_mobilenetv2.to(device)

    # Load the state dictionary of the trained MobileNetV2 fold model
    final_model_mobilenetv2.load_state_dict(torch.load(model_path_to_load_mobilenetv2, map_location=device))
    print(f"Successfully loaded MobileNetV2 model weights from: {model_path_to_load_mobilenetv2}")

    # Define the final save path for the consolidated MobileNetV2 model
    final_model_save_path_mobilenetv2 = os.path.join(
        gdrive_base_output_dir,
        f'{model_to_save_name_mobilenetv2}_final_model.pt'
    )

    # Save the state dictionary of the final_model_mobilenetv2
    torch.save(final_model_mobilenetv2.state_dict(), final_model_save_path_mobilenetv2)
    print(f"MobileNetV2 model saved successfully to: {final_model_save_path_mobilenetv2}")

    # Save all 5 folds training history for MobileNetV2
    if 'all_model_histories_mobilenetv2' in locals() and all_model_histories_mobilenetv2:
        history_save_path = os.path.join(
            gdrive_base_output_dir,
            f"{model_to_save_name_mobilenetv2}_all_folds_histories.json"
        )
        try:
            with open(history_save_path, 'w') as f:
                json.dump(all_model_histories_mobilenetv2, f, indent=4)
            print(f"All 5 folds training history for MobileNetV2 saved successfully to: {history_save_path}")
        except Exception as e:
            print(f"Error saving MobileNetV2 training histories: {e}")
    else:
        print("Warning: MobileNetV2 training histories not found in 'all_model_histories_mobilenetv2' variable.")

except FileNotFoundError:
    print(f"Error: MobileNetV2 model or config file not found at {model_path_to_load_mobilenetv2} or {fold_config_path_mobilenetv2}. Please ensure training was completed and the files exist.")
except Exception as e:
    print(f"Error saving MobileNetV2 model: {e}")

# %% [markdown] Cell 39
# Load the Trainnig Data

# %% Cell 40
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# The `plot_training_curves` function is defined in previous cells.
# We will re-define it here for clarity and robustness in case this cell is run independently.
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    # Adjust subplot creation for single or multiple folds
    if num_folds == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = [axes] # Make it iterable for consistent loop
    else:
        fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    fig.suptitle(f'Training and Validation Curves for {model_name}', fontsize=16)

    for i, history in enumerate(histories):
        # Plot Loss
        ax_loss = axes[i][0] if num_folds > 1 else axes[0]
        ax_loss.plot(history['train_loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'Fold {i+1} - Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # Plot Accuracy
        ax_acc = axes[i][1] if num_folds > 1 else axes[1]
        ax_acc.plot(history['train_acc'], label='Train Accuracy')
        ax_acc.plot(history['val_acc'], label='Validation Accuracy')
        ax_acc.set_title(f'Fold {i+1} - Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Define the path to the saved history file for MobileNetV2
# gdrive_base_output_dir is assumed to be defined from previous cells
model_name_mobilenetv2 = 'mobilenetv2_lung_clean'
load_history_path_mobilenetv2 = os.path.join(gdrive_base_output_dir, model_name_mobilenetv2 + "_all_folds_histories.json")

# Load the training history
try:
    with open(load_history_path_mobilenetv2, 'r') as f:
        loaded_histories_mobilenetv2 = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_mobilenetv2}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(loaded_histories_mobilenetv2[model_name_mobilenetv2])}")
    for i, history in enumerate(loaded_histories_mobilenetv2[model_name_mobilenetv2]):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(loaded_histories_mobilenetv2[model_name_mobilenetv2], model_name_mobilenetv2)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_mobilenetv2}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_mobilenetv2}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% Cell 41
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

set_all_seeds(SEED)

# Define the helper function for ensemble predictions
def get_ensemble_predictions_for_model_type(
    model_architecture,
    model_name_prefix,
    num_folds,
    base_output_dir_for_model,
    test_df,
    common_config,
    device,
    class_names
):
    print(f"\n--- Loading and collecting predictions for {model_name_prefix} (5-fold ensemble) ---")
    loaded_fold_models = {}

    for fold_number in range(1, num_folds + 1):
        model_save_dir = os.path.join(base_output_dir_for_model, model_name_prefix)
        fold_weight_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}.pt"
        )
        fold_config_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}_config.json"
        )

        try:
            with open(fold_config_path, "r") as f:
                fold_config = json.load(f)

            fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
            fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

            model = build_timm_model(
                model_architecture,
                fold_config
            )
            model.load_state_dict(
                torch.load(fold_weight_path, map_location=device)
            )
            model = model.to(device)
            model.eval()

            loaded_fold_models[f'fold_{fold_number}'] = {
                'model': model,
                'config': fold_config
            }
        except FileNotFoundError:
            print(f"Error: Weights or config not found for {model_name_prefix} Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
            continue
        except Exception as e:
            print(f"An error occurred while loading {model_name_prefix} Fold {fold_number}: {e}")
            continue

    if not loaded_fold_models:
        print(f"No models loaded for {model_name_prefix}. Cannot perform ensemble predictions.")
        return None, None, None

    all_fold_probs = []
    y_true_ret = None

    print(f"Collecting test predictions for {model_name_prefix}...")
    for fold_number in range(1, num_folds + 1):
        model_data = loaded_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            continue

        model = model_data['model']
        fold_config = model_data['config']

        transform_test = build_val_test_transform(fold_config)

        test_dataset = CustomImageDataset(
            test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=common_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"{model_name_prefix} Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)
        all_fold_probs.append(fold_probs)

        if y_true_ret is None:
            y_true_ret = fold_labels
        else:
            if not np.array_equal(y_true_ret, fold_labels):
                warnings.warn(f"Labels for {model_name_prefix} changed between folds, this should not happen with fixed random seed.")

    if not all_fold_probs:
        print(f"No predictions collected for {model_name_prefix}.")
        return None, None, None

    avg_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    return avg_probs, y_true_ret, class_names

model_name_mobilenetv2 = 'mobilenetv2_lung_clean'
model_to_load_architecture_mobilenetv2 = 'mobilenetv2_100'

# Get ensemble predictions for MobileNetV2
avg_probs_mobilenetv2, y_true_mobilenetv2, class_names_mobilenetv2 = get_ensemble_predictions_for_model_type(
    model_to_load_architecture_mobilenetv2,
    model_name_mobilenetv2,
    COMMON_CONFIG['num_folds'],
    gdrive_base_output_dir,
    clean_test_df,
    COMMON_CONFIG,
    device,
    class_names
)

if avg_probs_mobilenetv2 is not None:
    y_pred_mobilenetv2 = np.argmax(avg_probs_mobilenetv2, axis=1)

    # Calculate ensemble metrics
    acc_ensemble_mobilenetv2 = accuracy_score(y_true_mobilenetv2, y_pred_mobilenetv2)
    prec_ensemble_mobilenetv2 = precision_score(y_true_mobilenetv2, y_pred_mobilenetv2, average="weighted", zero_division=0)
    rec_ensemble_mobilenetv2 = recall_score(y_true_mobilenetv2, y_pred_mobilenetv2, average="weighted", zero_division=0)
    f1_ensemble_mobilenetv2 = f1_score(y_true_mobilenetv2, y_pred_mobilenetv2, average="weighted", zero_division=0)

    print("\n========== FINAL CLEAN TEST RESULT: MobileNetV2 5-Fold Ensemble ==========")
    print(f"Accuracy : {acc_ensemble_mobilenetv2:.4f}")
    print(f"Precision: {prec_ensemble_mobilenetv2:.4f}")
    print(f"Recall   : {rec_ensemble_mobilenetv2:.4f}")
    print(f"F1-score : {f1_ensemble_mobilenetv2:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true_mobilenetv2,
        y_pred_mobilenetv2,
        target_names=class_names_mobilenetv2,
        zero_division=0
    ))

    cm_ensemble_mobilenetv2 = confusion_matrix(
        y_true_mobilenetv2,
        y_pred_mobilenetv2,
        labels=range(len(class_names_mobilenetv2))
    )

    print("\nConfusion Matrix:")
    print(cm_ensemble_mobilenetv2)

    # Visualize the confusion matrix for the ensemble
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_ensemble_mobilenetv2,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names_mobilenetv2,
        yticklabels=class_names_mobilenetv2
    )
    plt.title("MobileNetV2 5-Fold Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Optionally save the ensemble results
    mobilenetv2_ensemble_output_file = os.path.join(
        gdrive_base_output_dir,
        'mobilenetv2_lung_clean_ensemble_test_results.json'
    )

    serializable_results_mobilenetv2 = {
        'accuracy': acc_ensemble_mobilenetv2,
        'precision': prec_ensemble_mobilenetv2,
        'recall': rec_ensemble_mobilenetv2,
        'f1_score': f1_ensemble_mobilenetv2,
        'confusion_matrix': cm_ensemble_mobilenetv2.tolist(),
        'class_names': class_names_mobilenetv2
    }

    try:
        with open(mobilenetv2_ensemble_output_file, 'w') as f:
            json.dump(serializable_results_mobilenetv2, f, indent=4)
        print(f"\nMobileNetV2 5-fold ensemble results saved to: {mobilenetv2_ensemble_output_file}")
    except Exception as e:
        print(f"An error occurred while saving MobileNetV2 ensemble results: {e}")

else:
    print("Could not perform MobileNetV2 ensemble evaluation as no model probabilities were collected.")

# %% Cell 42
import os
import json
import glob
import pandas as pd

# Base directory for outputs
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Find all ensemble result JSON files
ensemble_files = glob.glob(os.path.join(gdrive_base_output_dir, '*ensemble_test_results.json'))

ensemble_summary = []

print("Loading ensemble results...")
for file_path in ensemble_files:
    model_name = os.path.basename(file_path).replace('_ensemble_test_results.json', '')
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)

            ensemble_summary.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', None),
                'Precision': results.get('precision', None),
                'Recall': results.get('recall', None),
                'F1 Score': results.get('f1_score', None),
                'Macro AUC': results.get('macro_auc', 'N/A') # AUC might not be present in all files
            })
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Create a DataFrame for nice visualization
if ensemble_summary:
    summary_df = pd.DataFrame(ensemble_summary)
    # Sort by Accuracy in descending order
    summary_df.sort_values(by='Accuracy', ascending=False, inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    print("\n--- Ensemble Results Summary ---")
    display(summary_df)
else:
    print("No ensemble result files found in the output directory.")

# %% [markdown] Cell 43
# Densenet 121

# %% Cell 44
import os
import json

# Always use the clean no-leakage output folder
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

# DenseNet121 model specification
MODEL_SPECS_DENSENET121 = [
    {
        'model_to_load': 'densenet121',
        'model_name': 'densenet121_lung_clean',
    }
]

# Store histories
all_model_histories_densenet121 = {}

for model_spec in MODEL_SPECS_DENSENET121:
    model_name = model_spec['model_name']

    print("\n" + "=" * 60)
    print(f"Starting K-fold training for: {model_name}")
    print("=" * 60)

    histories = train_single_model_same_config(
        model_spec=model_spec,
        folds_info=folds_info,
        base_output_dir=gdrive_base_output_dir
    )

    all_model_histories_densenet121[model_name] = histories

print("\nDenseNet121 training completed.")
print("Saved model folder:")
print(os.path.join(gdrive_base_output_dir, 'densenet121_lung_clean'))

# %% Cell 45
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# The `plot_training_curves` function is defined in previous cells.
# We will re-define it here for clarity and robustness in case this cell is run independently.
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    # Adjust subplot creation for single or multiple folds
    if num_folds == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = [axes] # Make it iterable for consistent loop
    else:
        fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    fig.suptitle(f'Training and Validation Curves for {model_name}', fontsize=16)

    for i, history in enumerate(histories):
        # Plot Loss
        ax_loss = axes[i][0] if num_folds > 1 else axes[0]
        ax_loss.plot(history['train_loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'Fold {i+1} - Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # Plot Accuracy
        ax_acc = axes[i][1] if num_folds > 1 else axes[1]
        ax_acc.plot(history['train_acc'], label='Train Accuracy')
        ax_acc.plot(history['val_acc'], label='Validation Accuracy')
        ax_acc.set_title(f'Fold {i+1} - Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Define the path to the saved history file for DenseNet121
# gdrive_base_output_dir is assumed to be defined from previous cells
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
model_name_densenet = 'densenet121_lung_clean'
load_history_path_densenet = os.path.join(gdrive_base_output_dir, model_name_densenet + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_densenet, 'r') as f:
        loaded_histories_densenet = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_densenet}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(loaded_histories_densenet)}")
    for i, history in enumerate(loaded_histories_densenet):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(loaded_histories_densenet, model_name_densenet)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_densenet}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_densenet}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown] Cell 46
# Evaluate the classification

# %% Cell 47
import torch
import timm
import os
import json
import numpy as np

# Ensure gdrive_base_output_dir is defined correctly for the clean output folder
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

# Model details for DenseNet121
model_name_densenet121 = 'densenet121_lung_clean'
model_to_load_architecture_densenet121 = 'densenet121'

# Assuming the best performing fold's model or a specific fold's model is desired for loading
# For consistency with prior cells, we'll load the model from fold 5 as an example.
# In a real scenario, you might select the best performing fold based on validation metrics.
fold_number_to_load_densenet121 = 5

# Reconstruct the path where the model was saved during training
model_save_dir_densenet121 = os.path.join(gdrive_base_output_dir, model_name_densenet121)
model_path_to_load_densenet121 = os.path.join(
    model_save_dir_densenet121,
    f"{model_name_densenet121}_fold_{fold_number_to_load_densenet121}.pt"
)
fold_config_path_densenet121 = os.path.join(
    model_save_dir_densenet121,
    f"{model_name_densenet121}_fold_{fold_number_to_load_densenet121}_config.json"
)

# Load the fold-specific configuration for building the model
try:
    set_all_seeds(SEED)
    with open(fold_config_path_densenet121, 'r') as f:
        fold_config_densenet121 = json.load(f)

    # Convert mean/std back to numpy arrays (json stores them as lists)
    if 'mean_overall' in fold_config_densenet121 and isinstance(fold_config_densenet121['mean_overall'], list):
        fold_config_densenet121['mean_overall'] = np.array(fold_config_densenet121['mean_overall'])
    if 'std_overall' in fold_config_densenet121 and isinstance(fold_config_densenet121['std_overall'], list):
        fold_config_densenet121['std_overall'] = np.array(fold_config_densenet121['std_overall'])

    # Build the DenseNet121 model architecture
    final_model_densenet121 = build_timm_model(model_to_load_architecture_densenet121, fold_config_densenet121)
    final_model_densenet121 = final_model_densenet121.to(device)

    # Load the state dictionary of the trained DenseNet121 fold model
    final_model_densenet121.load_state_dict(torch.load(model_path_to_load_densenet121, map_location=device))
    print(f"Successfully loaded DenseNet121 model weights from: {model_path_to_load_densenet121}")

    # Ensure the model is in evaluation mode
    final_model_densenet121.eval()

    print("\nEvaluating DenseNet121 on the final test set...")

    # Prepare configuration for the test set evaluation
    # Use mean/std calculated from the clean_train_val_df for consistent normalization
    test_mean_overall_densenet121, test_std_overall_densenet121 = compute_mean_std_from_dataframe(clean_train_val_df)

    test_evaluation_config_densenet121 = COMMON_CONFIG.copy()
    test_evaluation_config_densenet121['mean_overall'] = test_mean_overall_densenet121
    test_evaluation_config_densenet121['std_overall'] = test_std_overall_densenet121

    # Build test transformations
    transform_test_densenet121 = build_val_test_transform(test_evaluation_config_densenet121)

    # Create CustomImageDataset and DataLoader for the test set
    test_dataset_densenet121 = CustomImageDataset(
        clean_test_df,
        transform=transform_test_densenet121,
        labels_map=test_evaluation_config_densenet121['labels_map']
    )

    test_loader_densenet121 = DataLoader(
        test_dataset_densenet121,
        batch_size=test_evaluation_config_densenet121['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Evaluate the model on the test set
    test_results_densenet121 = evaluate_model(
        model=final_model_densenet121,
        dataloader=test_loader_densenet121,
        device=device,
        class_names=test_evaluation_config_densenet121['class_names'],
        title="DenseNet121 Final Test Set Evaluation",
        show_plots=True # Set to False if you don't want plots immediately
    )

    # Print the cumulative accuracy
    cumulative_accuracy_densenet121 = test_results_densenet121['accuracy']
    print(f"\nCumulative Accuracy of DenseNet121 on the test set: {cumulative_accuracy_densenet121:.4f}")

    # Optionally, save the test results to a JSON file
    output_file_path_densenet121 = os.path.join(
        gdrive_base_output_dir,
        'densenet121_lung_clean_cumulative_test_results.json'
    )

    serializable_test_results_densenet121 = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in test_results_densenet121.items() if k not in ['labels', 'preds', 'probs']
    }

    try:
        with open(output_file_path_densenet121, 'w') as f:
            json.dump(serializable_test_results_densenet121, f, indent=4)
        print(f"DenseNet121 test results saved to: {output_file_path_densenet121}")
    except Exception as e:
        print(f"An error occurred while saving DenseNet121 test results: {e}")

except FileNotFoundError:
    print(f"Error: DenseNet121 model or config file not found at {model_path_to_load_densenet121} or {fold_config_path_densenet121}. Please ensure training was completed and the files exist.")
except Exception as e:
    print(f"An error occurred while loading or evaluating the DenseNet121 model: {e}")

# %% [markdown] Cell 48
# ensamble accuracy

# %% Cell 49
import torch
import timm
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

set_all_seeds(SEED)

# Define a helper function to get 5-fold ensemble probabilities for a given model type
def get_ensemble_predictions_for_model_type(
    model_architecture,
    model_name_prefix,
    num_folds,
    base_output_dir_for_model,
    test_df,
    common_config,
    device,
    class_names
):
    print(f"\n--- Loading and collecting predictions for {model_name_prefix} (5-fold ensemble) ---")
    loaded_fold_models = {}

    for fold_number in range(1, num_folds + 1):
        model_save_dir = os.path.join(base_output_dir_for_model, model_name_prefix)
        fold_weight_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}.pt"
        )
        fold_config_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}_config.json"
        )

        try:
            with open(fold_config_path, "r") as f:
                fold_config = json.load(f)

            # Convert mean/std back to numpy arrays if they were stored as lists
            fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
            fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

            model = build_timm_model(
                model_architecture,
                fold_config # Pass the fold-specific config
            )
            model.load_state_dict(
                torch.load(fold_weight_path, map_location=device)
            )
            model = model.to(device)
            model.eval() # Set model to evaluation mode

            loaded_fold_models[f'fold_{fold_number}'] = {
                'model': model,
                'config': fold_config
            }
            # print(f"{model_name_prefix} Fold {fold_number} loaded successfully.")

        except FileNotFoundError:
            print(f"Error: Weights or config not found for {model_name_prefix} Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
            continue # Skip this fold if files are missing
        except Exception as e:
            print(f"An error occurred while loading {model_name_prefix} Fold {fold_number}: {e}")
            continue # Skip this fold on other errors

    if not loaded_fold_models:
        print(f"No models loaded for {model_name_prefix}. Cannot perform ensemble predictions.")
        return None, None, None

    all_fold_probs = []
    y_true_ret = None

    print(f"Collecting test predictions for {model_name_prefix}...")
    for fold_number in range(1, num_folds + 1):
        model_data = loaded_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            continue # Already printed error during loading if model was missing

        model = model_data['model']
        fold_config = model_data['config']

        transform_test = build_val_test_transform(fold_config)

        test_dataset = CustomImageDataset(
            test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=common_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"{model_name_prefix} Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)
        all_fold_probs.append(fold_probs)

        if y_true_ret is None:
            y_true_ret = fold_labels
        else:
            if not np.array_equal(y_true_ret, fold_labels):
                warnings.warn(f"Labels for {model_name_prefix} changed between folds, this should not happen with fixed random seed.")

    if not all_fold_probs:
        print(f"No predictions collected for {model_name_prefix}.")
        return None, None, None

    avg_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    return avg_probs, y_true_ret, class_names


# Model details for DenseNet121
model_name_densenet121 = 'densenet121_lung_clean'
model_to_load_architecture_densenet121 = 'densenet121'

# Ensure gdrive_base_output_dir is defined
# (It should be from previous cells, but defining it here for robustness)
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Get ensemble predictions for DenseNet121
avg_probs_densenet121, y_true_densenet121, class_names_densenet121 = get_ensemble_predictions_for_model_type(
    model_to_load_architecture_densenet121,
    model_name_densenet121,
    COMMON_CONFIG['num_folds'],
    gdrive_base_output_dir,
    clean_test_df,
    COMMON_CONFIG,
    device,
    class_names
)

if avg_probs_densenet121 is not None:
    y_pred_densenet121 = np.argmax(avg_probs_densenet121, axis=1)

    # Calculate ensemble metrics
    acc_ensemble_densenet121 = accuracy_score(y_true_densenet121, y_pred_densenet121)
    prec_ensemble_densenet121 = precision_score(y_true_densenet121, y_pred_densenet121, average="weighted", zero_division=0)
    rec_ensemble_densenet121 = recall_score(y_true_densenet121, y_pred_densenet121, average="weighted", zero_division=0)
    f1_ensemble_densenet121 = f1_score(y_true_densenet121, y_pred_densenet121, average="weighted", zero_division=0)

    print("\n========== FINAL CLEAN TEST RESULT: DenseNet121 5-Fold Ensemble ==========")
    print(f"Accuracy : {acc_ensemble_densenet121:.4f}")
    print(f"Precision: {prec_ensemble_densenet121:.4f}")
    print(f"Recall   : {rec_ensemble_densenet121:.4f}")
    print(f"F1-score : {f1_ensemble_densenet121:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true_densenet121,
        y_pred_densenet121,
        target_names=class_names_densenet121,
        zero_division=0
    ))

    cm_ensemble_densenet121 = confusion_matrix(
        y_true_densenet121,
        y_pred_densenet121,
        labels=range(len(class_names_densenet121))
    )

    print("\nConfusion Matrix:")
    print(cm_ensemble_densenet121)

    # Visualize the confusion matrix for the ensemble
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_ensemble_densenet121,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names_densenet121,
        yticklabels=class_names_densenet121
    )
    plt.title("DenseNet121 5-Fold Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Optionally save the ensemble results
    densenet121_ensemble_output_file = os.path.join(
        gdrive_base_output_dir,
        'densenet121_lung_clean_ensemble_test_results.json'
    )

    serializable_results_densenet121 = {
        'accuracy': acc_ensemble_densenet121,
        'precision': prec_ensemble_densenet121,
        'recall': rec_ensemble_densenet121,
        'f1_score': f1_ensemble_densenet121,
        'confusion_matrix': cm_ensemble_densenet121.tolist(),
        'class_names': class_names_densenet121
    }

    try:
        with open(densenet121_ensemble_output_file, 'w') as f:
            json.dump(serializable_results_densenet121, f, indent=4)
        print(f"\nDenseNet121 5-fold ensemble results saved to: {densenet121_ensemble_output_file}")
    except Exception as e:
        print(f"An error occurred while saving DenseNet121 ensemble results: {e}")

else:
    print("Could not perform DenseNet121 ensemble evaluation as no model probabilities were collected.")

# %% [markdown] Cell 50
# Load the result

# %% Cell 51
import json
import os

filepath = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE/densenet121_lung_clean_ensemble_test_results.json'

if os.path.exists(filepath):
    with open(filepath, 'r') as f:
        densenet121_results = json.load(f)
    print("DenseNet121 5-fold ensemble results:")
    # Printing the results nicely formatted, excluding the confusion matrix for brevity if it's too large,
    # but here we'll just print the main metrics.
    for key, value in densenet121_results.items():
        if key != 'confusion_matrix' and key != 'class_names':
            print(f"{key}: {value}")
    print("\nClass Names:", densenet121_results.get('class_names'))
    print("Confusion Matrix:", densenet121_results.get('confusion_matrix'))
else:
    print(f"File not found: {filepath}")

# %% [markdown] Cell 52
# Dense201

# %% Cell 53
import os
import json

# IMPORTANT: use only the clean no-leakage output folder
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

# DenseNet201 model specification
MODEL_SPECS_DENSENET201 = [
    {
        'model_to_load': 'densenet201',
        'model_name': 'densenet201_lung_clean',
    }
]

# Store DenseNet201 histories
all_model_histories_densenet201 = {}

for model_spec in MODEL_SPECS_DENSENET201:
    model_name = model_spec['model_name']

    print("\n" + "=" * 60)
    print(f"Starting K-fold training for: {model_name}")
    print("=" * 60)

    histories = train_single_model_same_config(
        model_spec=model_spec,
        folds_info=folds_info,
        base_output_dir=gdrive_base_output_dir
    )

    all_model_histories_densenet201[model_name] = histories

print("\nDenseNet201 training completed.")
print("Saved model folder:")
print(os.path.join(gdrive_base_output_dir, 'densenet201_lung_clean'))

# %% Cell 54
import torch
import timm
import os
import json
import numpy as np

# Ensure gdrive_base_output_dir is defined correctly for the clean output folder
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

# Model details for DenseNet201
model_name_densenet201 = 'densenet201_lung_clean'
model_to_load_architecture_densenet201 = 'densenet201'

# Assuming the best performing fold's model or a specific fold's model is desired for loading
# For consistency with prior cells, we'll load the model from fold 5 as an example.
# In a real scenario, you might select the best performing fold based on validation metrics.
fold_number_to_load_densenet201 = 5

# Reconstruct the path where the model was saved during training
model_save_dir_densenet201 = os.path.join(gdrive_base_output_dir, model_name_densenet201)
model_path_to_load_densenet201 = os.path.join(
    model_save_dir_densenet201,
    f"{model_name_densenet201}_fold_{fold_number_to_load_densenet201}.pt"
)

# Instantiate the model architecture (using the build_timm_model function from a previous cell)
# It's important to use the config that was used for training this specific model.
model_build_config_densenet201 = COMMON_CONFIG.copy()
model_build_config_densenet201['ImageNet'] = True # DenseNet201 usually uses ImageNet pretraining
model_build_config_densenet201['num_classes'] = num_classes # Make sure num_classes is defined

# Build the model
final_model_densenet201 = build_timm_model(model_to_load_architecture_densenet201, model_build_config_densenet201)
final_model_densenet201 = final_model_densenet201.to(device)

# Load the state dictionary
try:
    final_model_densenet201.load_state_dict(torch.load(model_path_to_load_densenet201, map_location=device))
    print(f"Successfully loaded DenseNet201 model weights from: {model_path_to_load_densenet201}")

    # Ensure the model is in evaluation mode
    final_model_densenet201.eval()

    print("\nEvaluating DenseNet201 on the final test set...")

    # Prepare configuration for the test set evaluation
    # Use mean/std calculated from the clean_train_val_df for consistent normalization
    test_mean_overall_densenet201, test_std_overall_densenet201 = compute_mean_std_from_dataframe(clean_train_val_df)

    test_evaluation_config_densenet201 = COMMON_CONFIG.copy()
    test_evaluation_config_densenet201['mean_overall'] = test_mean_overall_densenet201
    test_evaluation_config_densenet201['std_overall'] = test_std_overall_densenet201

    # Build test transformations
    transform_test_densenet201 = build_val_test_transform(test_evaluation_config_densenet201)

    # Create CustomImageDataset and DataLoader for the test set
    test_dataset_densenet201 = CustomImageDataset(
        clean_test_df,
        transform=transform_test_densenet201,
        labels_map=test_evaluation_config_densenet201['labels_map']
    )

    test_loader_densenet201 = DataLoader(
        test_dataset_densenet201,
        batch_size=test_evaluation_config_densenet201['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Evaluate the model on the test set
    test_results_densenet201 = evaluate_model(
        model=final_model_densenet201,
        dataloader=test_loader_densenet201,
        device=device,
        class_names=test_evaluation_config_densenet201['class_names'],
        title="DenseNet201 Final Test Set Evaluation",
        show_plots=True # Set to False if you don't want plots immediately
    )

    # Print the cumulative accuracy
    cumulative_accuracy_densenet201 = test_results_densenet201['accuracy']
    print(f"\nCumulative Accuracy of DenseNet201 on the test set: {cumulative_accuracy_densenet201:.4f}")

    # Load and plot training histories for DenseNet201
    load_history_path_densenet201 = os.path.join(gdrive_base_output_dir, model_name_densenet201 + "_histories_clean.json")

    try:
        with open(load_history_path_densenet201, 'r') as f:
            loaded_histories_densenet201 = json.load(f)
        print(f"\nSuccessfully loaded training histories from: {load_history_path_densenet201}")

        # Display some information from the loaded histories
        print(f"\nNumber of folds in history: {len(loaded_histories_densenet201)}")
        for i, history in enumerate(loaded_histories_densenet201):
            print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
            print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
            print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
            if 'lr' in history and len(history['lr']) > 0:
                print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

        # Plot the training curves after loading histories
        plot_training_curves(loaded_histories_densenet201, model_name_densenet201)

    except FileNotFoundError:
        print(f"Error: History file not found at {load_history_path_densenet201}. Please ensure training was completed and the file was saved.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {load_history_path_densenet201}. The file might be corrupted or empty.")
    except Exception as e:
        print(f"An unexpected error occurred while loading histories: {e}")

except FileNotFoundError:
    print(f"Error: DenseNet201 model file not found at {model_path_to_load_densenet201}. Please ensure the training completed successfully and the file exists.")
except Exception as e:
    print(f"An error occurred while loading or evaluating the DenseNet201 model: {e}")

# %% [markdown] Cell 55
# loaded  model

# %% Cell 56
import torch
import timm
import os
import json
import numpy as np

# Model details for DenseNet201
model_name_densenet201 = 'densenet201_lung_clean'
model_to_load_densenet201 = 'densenet201'

# Dictionary to store all loaded models and their configs
loaded_densenet201_fold_models = {}

print(f"Loading all {COMMON_CONFIG['num_folds']} folds for {model_name_densenet201}...")

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    model_save_dir = os.path.join(gdrive_base_output_dir, model_name_densenet201)

    fold_weight_path = os.path.join(
        model_save_dir,
        f"{model_name_densenet201}_fold_{fold_number}.pt"
    )

    fold_config_path = os.path.join(
        model_save_dir,
        f"{model_name_densenet201}_fold_{fold_number}_config.json"
    )

    try:
        with open(fold_config_path, "r") as f:
            fold_config = json.load(f)

        fold_config['mean_overall'] = np.array(fold_config['mean_overall'])
        fold_config['std_overall'] = np.array(fold_config['std_overall'])

        model_densenet201 = timm.create_model(
            model_to_load_densenet201,
            pretrained=False, # Should be False since we are loading custom trained weights
            num_classes=fold_config['num_classes']
        )

        model_densenet201.load_state_dict(
            torch.load(fold_weight_path, map_location=device)
        )

        model_densenet201 = model_densenet201.to(device)
        model_densenet201.eval() # Set model to evaluation mode

        loaded_densenet201_fold_models[f'fold_{fold_number}'] = {
            'model': model_densenet201,
            'config': fold_config,
            'weight_path': fold_weight_path
        }

        print(f"DenseNet201 Fold {fold_number} loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Weights or config not found for Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
    except Exception as e:
        print(f"An error occurred while loading DenseNet201 Fold {fold_number}: {e}")

print(f"\nAll {len(loaded_densenet201_fold_models)} DenseNet201 models loaded.")
if loaded_densenet201_fold_models:
    print("Example: Accessing model for Fold 1:")
    print(loaded_densenet201_fold_models['fold_1']['model'])
    print("Example: Accessing config for Fold 1:")
    print(loaded_densenet201_fold_models['fold_1']['config'])

# %% [markdown] Cell 57
# load the model

# %% Cell 58
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# The `plot_training_curves` function is defined in a previous cell.
# We will ensure it's available here for clarity if this cell is run independently.
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    if num_folds == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    fig.suptitle(f'Training and Validation Curves for {model_name}', fontsize=16)

    for i, history in enumerate(histories):
        ax_loss = axes[i][0] if num_folds > 1 else axes[0]
        ax_loss.plot(history['train_loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'Fold {i+1} - Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        ax_acc = axes[i][1] if num_folds > 1 else axes[1]
        ax_acc.plot(history['train_acc'], label='Train Accuracy')
        ax_acc.plot(history['val_acc'], label='Validation Accuracy')
        ax_acc.set_title(f'Fold {i+1} - Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Define the path to the saved history file for DenseNet201
# gdrive_base_output_dir is assumed to be defined from previous cells
model_name_densenet201 = 'densenet201_lung_clean'
load_history_path_densenet201 = os.path.join(gdrive_base_output_dir, model_name_densenet201 + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_densenet201, 'r') as f:
        loaded_histories_densenet201 = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_densenet201}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(loaded_histories_densenet201)}")
    for i, history in enumerate(loaded_histories_densenet201):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(loaded_histories_densenet201, model_name_densenet201)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_densenet201}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_densenet201}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown] Cell 59
# Ensamble Accuracy

# %% Cell 60
import torch
import timm
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

set_all_seeds(SEED)

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Model details for DenseNet201
model_name_densenet201 = 'densenet201_lung_clean'
model_to_load_architecture_densenet201 = 'densenet201'

# Dictionary to store all loaded models and their configs
loaded_densenet201_fold_models = {}

print(f"Loading all {COMMON_CONFIG['num_folds']} folds for {model_name_densenet201}...")

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    model_save_dir = os.path.join(gdrive_base_output_dir, model_name_densenet201)

    fold_weight_path = os.path.join(
        model_save_dir,
        f"{model_name_densenet201}_fold_{fold_number}.pt"
    )

    fold_config_path = os.path.join(
        model_save_dir,
        f"{model_name_densenet201}_fold_{fold_number}_config.json"
    )

    try:
        with open(fold_config_path, "r") as f:
            fold_config = json.load(f)

        # Convert mean/std back to numpy arrays
        fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
        fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

        model_densenet201 = timm.create_model(
            model_to_load_architecture_densenet201,
            pretrained=False, # We are loading custom trained weights
            num_classes=fold_config['num_classes']
        )

        model_densenet201.load_state_dict(
            torch.load(fold_weight_path, map_location=device)
        )

        model_densenet201 = model_densenet201.to(device)
        model_densenet201.eval() # Set model to evaluation mode

        loaded_densenet201_fold_models[f'fold_{fold_number}'] = {
            'model': model_densenet201,
            'config': fold_config
        }

        print(f"DenseNet201 Fold {fold_number} loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Weights or config not found for Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
    except Exception as e:
        print(f"An error occurred while loading DenseNet201 Fold {fold_number}: {e}")

print(f"\nAll {len(loaded_densenet201_fold_models)} DenseNet201 models loaded.")

# Prepare for ensemble prediction
all_fold_probs_densenet201 = []
y_true_final_densenet201 = None
class_names_densenet201 = None

if loaded_densenet201_fold_models:
    print(f"\nEvaluating all {COMMON_CONFIG['num_folds']} DenseNet201 fold models on the final test set for ensemble...")

    for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
        print(f"\n--- Collecting predictions for DenseNet201 Fold {fold_number} ---")

        model_data = loaded_densenet201_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            print(f"Skipping Fold {fold_number} due to missing model data.")
            continue

        model = model_data['model']
        fold_config = model_data['config']

        # Build test transformations using the fold-specific mean/std
        transform_test = build_val_test_transform(fold_config)

        # Create CustomImageDataset and DataLoader for the test set
        test_dataset = CustomImageDataset(
            clean_test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"DenseNet201 Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        all_fold_probs_densenet201.append(fold_probs)

        if y_true_final_densenet201 is None:
            y_true_final_densenet201 = fold_labels
            class_names_densenet201 = fold_config['class_names']
        else:
            assert np.array_equal(y_true_final_densenet201, fold_labels), "Label order changed between folds!"

    if all_fold_probs_densenet201:
        # Average the probabilities across all folds
        avg_probs_densenet201 = np.mean(np.stack(all_fold_probs_densenet201, axis=0), axis=0)
        y_pred_final_densenet201 = np.argmax(avg_probs_densenet201, axis=1)

        # Calculate ensemble metrics
        acc_ensemble_densenet201 = accuracy_score(y_true_final_densenet201, y_pred_final_densenet201)
        prec_ensemble_densenet201 = precision_score(y_true_final_densenet201, y_pred_final_densenet201, average="weighted", zero_division=0)
        rec_ensemble_densenet201 = recall_score(y_true_final_densenet201, y_pred_final_densenet201, average="weighted", zero_division=0)
        f1_ensemble_densenet201 = f1_score(y_true_final_densenet201, y_pred_final_densenet201, average="weighted", zero_division=0)

        print("\n========== FINAL CLEAN TEST RESULT: DenseNet201 5-Fold Ensemble ==========")
        print(f"Accuracy : {acc_ensemble_densenet201:.4f}")
        print(f"Precision: {prec_ensemble_densenet201:.4f}")
        print(f"Recall   : {rec_ensemble_densenet201:.4f}")
        print(f"F1-score : {f1_ensemble_densenet201:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_true_final_densenet201,
            y_pred_final_densenet201,
            target_names=class_names_densenet201,
            zero_division=0
        ))

        cm_ensemble_densenet201 = confusion_matrix(
            y_true_final_densenet201,
            y_pred_final_densenet201,
            labels=range(len(class_names_densenet201))
        )

        print("\nConfusion Matrix:")
        print(cm_ensemble_densenet201)
    else:
        print("No DenseNet201 fold probabilities were collected.")
else:
    print("No DenseNet201 models were loaded to perform ensemble evaluation.")

# %% Cell 61
import json
import os
from sklearn.metrics import roc_auc_score

# Calculate Macro AUC for the DenseNet201 Model
try:
    macro_auc_densenet201 = roc_auc_score(
        y_true_final_densenet201,
        avg_probs_densenet201,
        multi_class='ovr',
        average='macro'
    )
    print(f"DenseNet201 Macro AUC: {macro_auc_densenet201:.4f}\n")
except Exception as e:
    macro_auc_densenet201 = None
    print(f"Could not calculate AUC: {e}")

# Save the ensemble results for DenseNet201
densenet201_ensemble_output_file = os.path.join(
    gdrive_base_output_dir,
    'densenet201_lung_clean_ensemble_test_results.json'
)

serializable_results_densenet201 = {
    'accuracy': acc_ensemble_densenet201,
    'precision': prec_ensemble_densenet201,
    'recall': rec_ensemble_densenet201,
    'f1_score': f1_ensemble_densenet201,
    'macro_auc': macro_auc_densenet201,
    'confusion_matrix': cm_ensemble_densenet201.tolist(),
    'class_names': class_names_densenet201
}

try:
    with open(densenet201_ensemble_output_file, 'w') as f:
        json.dump(serializable_results_densenet201, f, indent=4)
    print(f"DenseNet201 5-fold ensemble results successfully saved to:\n{densenet201_ensemble_output_file}")
except Exception as e:
    print(f"An error occurred while saving DenseNet201 ensemble results: {e}")

# %% Cell 62
import json
import os

filepath_d201 = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE/densenet201_lung_clean_ensemble_test_results.json'

if os.path.exists(filepath_d201):
    with open(filepath_d201, 'r') as f:
        densenet201_results = json.load(f)
    print("DenseNet201 5-fold ensemble results:")
    for key, value in densenet201_results.items():
        if key not in ['confusion_matrix', 'class_names']:
            print(f"{key}: {value}")
    print("\nClass Names:", densenet201_results.get('class_names'))
    print("Confusion Matrix:", densenet201_results.get('confusion_matrix'))
else:
    print(f"File not found: {filepath_d201}")

# %% [markdown] Cell 63
# GRAD CAM

# %% Cell 64
# ============================================================
# DenseNet201 Grad-CAM - All Classes / One Test Image
# ============================================================

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

# Notebook command: !pip install -q timm opencv-python

import os
import json
import cv2
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# SETTINGS
# ============================================================

BASE_OUTPUT_DIR = "/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE"

MODEL_NAME = "densenet201_lung_clean"
MODEL_TO_LOAD = "densenet201"

# Choose one fold for Grad-CAM
FOLD = 1

# Change this index to visualize another image from the saved final test set
IMAGE_INDEX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, MODEL_NAME)

weight_path = os.path.join(
    MODEL_DIR,
    f"{MODEL_NAME}_fold_{FOLD}.pt"
)

config_path = os.path.join(
    MODEL_DIR,
    f"{MODEL_NAME}_fold_{FOLD}_config.json"
)

test_csv_path = os.path.join(
    BASE_OUTPUT_DIR,
    "clean_final_test_split.csv"
)

print("Weight exists:", os.path.exists(weight_path))
print("Config exists:", os.path.exists(config_path))
print("Test CSV exists:", os.path.exists(test_csv_path))

# ============================================================
# LOAD CONFIG AND TEST CSV
# ============================================================

with open(config_path, "r") as f:
    fold_config = json.load(f)

fold_config["mean_overall"] = np.array(fold_config["mean_overall"])
fold_config["std_overall"] = np.array(fold_config["std_overall"])

class_names = fold_config["class_names"]
labels_map = fold_config["labels_map"]
idx_to_class = {v: k for k, v in labels_map.items()}

clean_test_df = pd.read_csv(test_csv_path)

print("\nClass names:", class_names)
print("Total test images:", len(clean_test_df))

# ============================================================
# TRANSFORM
# ============================================================

def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

test_transform = transforms.Compose([
    transforms.Resize((fold_config["Resize_h"], fold_config["Resize_w"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=as_list(fold_config["mean_overall"]),
        std=as_list(fold_config["std_overall"])
    )
])

# ============================================================
# LOAD IMAGE
# ============================================================

img_path = clean_test_df.iloc[IMAGE_INDEX]["filepath"]
true_label_name = clean_test_df.iloc[IMAGE_INDEX]["label"]
true_label_idx = labels_map[true_label_name]

pil_img = Image.open(img_path).convert("RGB")
input_tensor = test_transform(pil_img).unsqueeze(0).to(device)

# For visualization: resized original image without normalization
vis_img = pil_img.resize((fold_config["Resize_w"], fold_config["Resize_h"]))
vis_img_np = np.array(vis_img).astype(np.float32) / 255.0

print("\nImage path:", img_path)
print("True label:", true_label_name)

# ============================================================
# LOAD DENSENET201 MODEL
# ============================================================

model = timm.create_model(
    MODEL_TO_LOAD,
    pretrained=False,
    num_classes=fold_config["num_classes"]
)

state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print("DenseNet201 model loaded successfully.")

# ============================================================
# TARGET LAYER FOR DENSENET201
# ============================================================

# Good target layer for timm DenseNet201
if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
    target_layer = model.features.denseblock4
    print("Target layer: model.features.denseblock4")
elif hasattr(model, "features") and hasattr(model.features, "norm5"):
    target_layer = model.features.norm5
    print("Target layer: model.features.norm5")
else:
    print(model)
    raise ValueError("Could not find target layer. Choose the final convolutional layer manually.")

# ============================================================
# CUSTOM GRAD-CAM CLASS
# ============================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0, pred_idx].item()

        if class_idx is None:
            class_idx = pred_idx

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, pred_idx, pred_prob

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

# ============================================================
# GENERATE GRAD-CAM FOR ALL CLASSES
# ============================================================

# First get overall prediction
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    overall_pred_idx = torch.argmax(probs, dim=1).item()
    overall_pred_prob = probs[0, overall_pred_idx].item()
    overall_pred_label_name = idx_to_class[overall_pred_idx]

print("\nPredicted label:", overall_pred_label_name)
print(f"Prediction confidence: {overall_pred_prob:.4f}")

num_classes = fold_config["num_classes"]
fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))

# Plot original image
axes[0].imshow(vis_img_np)
axes[0].set_title(f"Original\nTrue: {true_label_name}\nPred: {overall_pred_label_name}")
axes[0].axis("off")

for class_idx in range(num_classes):
    cam_generator = GradCAM(model, target_layer)
    cam, _, _ = cam_generator.generate(input_tensor, class_idx=class_idx)
    cam_generator.remove_hooks()

    class_name = idx_to_class[class_idx]

    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = 0.55 * vis_img_np + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # Plot overlay
    axes[class_idx + 1].imshow(overlay)
    title_str = f"Grad-CAM: {class_name}"
    if class_idx == overall_pred_idx:
        title_str += "\n(Predicted)"
    if class_idx == true_label_idx:
        title_str += "\n(True)"
    axes[class_idx + 1].set_title(title_str)
    axes[class_idx + 1].axis("off")

plt.tight_layout()

# ============================================================
# SAVE GRAD-CAM OUTPUT
# ============================================================

save_dir = os.path.join(BASE_OUTPUT_DIR, "gradcam_outputs")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(
    save_dir,
    f"gradcam_all_classes_{MODEL_NAME}_fold_{FOLD}_image_{IMAGE_INDEX}.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("Grad-CAM saved to:", save_path)

# %% [markdown] Cell 65
# MobilenetV3 large 100

# %% Cell 66
# ==============================
# Train MobileNetV3 Large 100 with same setup
# ==============================

import os
import json
import torch
import timm
import numpy as np

# Always use clean no-leakage output folder
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
os.makedirs(gdrive_base_output_dir, exist_ok=True)

# MobileNetV3 Large 100 model specification
MODEL_SPECS_MOBILENETV3_LARGE = [
    {
        'model_to_load': 'mobilenetv3_large_100',
        'model_name': 'mobilenetv3_large_100_lung_clean',
    }
]

# Store histories
all_model_histories_mobilenetv3_large = {}

for model_spec in MODEL_SPECS_MOBILENETV3_LARGE:
    model_name = model_spec['model_name']

    print("\n" + "=" * 60)
    print(f"Starting K-fold training for: {model_name}")
    print("=" * 60)

    histories = train_single_model_same_config(
        model_spec=model_spec,
        folds_info=folds_info,
        base_output_dir=gdrive_base_output_dir
    )

    all_model_histories_mobilenetv3_large[model_name] = histories

print("\nMobileNetV3 Large 100 training completed.")
print("Saved model folder:")
print(os.path.join(gdrive_base_output_dir, 'mobilenetv3_large_100_lung_clean'))

# %% [markdown] Cell 67
# load the model

# %% Cell 68
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# The `plot_training_curves` function is defined in a previous cell.
# Ensuring it's available in this scope for clarity if this cell is run independently.
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    # Adjust subplot creation for single or multiple folds
    if num_folds == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = [axes] # Make it iterable for consistent loop
    else:
        fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    fig.suptitle(f'Training and Validation Curves for {model_name}', fontsize=16)

    for i, history in enumerate(histories):
        # Plot Loss
        ax_loss = axes[i][0] if num_folds > 1 else axes[0]
        ax_loss.plot(history['train_loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'Fold {i+1} - Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # Plot Accuracy
        ax_acc = axes[i][1] if num_folds > 1 else axes[1]
        ax_acc.plot(history['train_acc'], label='Train Accuracy')
        ax_acc.plot(history['val_acc'], label='Validation Accuracy')
        ax_acc.set_title(f'Fold {i+1} - Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Define the path to the saved history file for MobileNetV3 Large
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
model_name_mobilenetv3_large = 'mobilenetv3_large_100_lung_clean'
load_history_path_mobilenetv3_large = os.path.join(gdrive_base_output_dir, model_name_mobilenetv3_large + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_mobilenetv3_large, 'r') as f:
        loaded_histories_mobilenetv3_large = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_mobilenetv3_large}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(loaded_histories_mobilenetv3_large)}")
    for i, history in enumerate(loaded_histories_mobilenetv3_large):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(loaded_histories_mobilenetv3_large, model_name_mobilenetv3_large)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_mobilenetv3_large}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_mobilenetv3_large}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown] Cell 69
# Average accuary

# %% Cell 70
import torch
import os
import timm

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Model details for mobilenetv3_large_100
model_to_save_name_mobilenetv3_large = 'mobilenetv3_large_100_lung_clean'
model_to_load_architecture_mobilenetv3_large = 'mobilenetv3_large_100'

# Assuming the last fold's model is desired for loading
# Adjust the fold number if a specific fold's best model is preferred.
fold_number_to_load_mobilenetv3_large = 5 # Example: loading model from fold 5

# Reconstruct the path where the model was saved during training
model_save_dir_mobilenetv3_large = os.path.join(gdrive_base_output_dir, model_to_save_name_mobilenetv3_large)
model_path_to_load_mobilenetv3_large = os.path.join(
    model_save_dir_mobilenetv3_large,
    f"{model_to_save_name_mobilenetv3_large}_fold_{fold_number_to_load_mobilenetv3_large}.pt"
)

# Instantiate the model architecture (using the build_timm_model function from a previous cell)
# It's important to use the config that was used for training this specific model.
model_build_config_mobilenetv3_large = COMMON_CONFIG.copy()
model_build_config_mobilenetv3_large['ImageNet'] = True # MobileNetV3 usually uses ImageNet pretraining
model_build_config_mobilenetv3_large['num_classes'] = num_classes # Make sure num_classes is defined

final_model_mobilenetv3_large = build_timm_model(model_to_load_architecture_mobilenetv3_large, model_build_config_mobilenetv3_large)
final_model_mobilenetv3_large = final_model_mobilenetv3_large.to(device)

# Load the state dictionary
try:
    final_model_mobilenetv3_large.load_state_dict(torch.load(model_path_to_load_mobilenetv3_large, map_location=device))
    print(f"Successfully loaded MobileNetV3 Large model weights from: {model_path_to_load_mobilenetv3_large}")

    # Ensure the model is in evaluation mode
    final_model_mobilenetv3_large.eval()

    print("\nEvaluating MobileNetV3 Large on the final test set...")

    # Prepare configuration for the test set evaluation
    # Use mean/std calculated from the clean_train_val_df for consistent normalization
    test_mean_overall, test_std_overall = compute_mean_std_from_dataframe(clean_train_val_df)

    test_evaluation_config_mobilenetv3_large = COMMON_CONFIG.copy()
    test_evaluation_config_mobilenetv3_large['mean_overall'] = test_mean_overall
    test_evaluation_config_mobilenetv3_large['std_overall'] = test_std_overall

    # Build test transformations
    transform_test_mobilenetv3_large = build_val_test_transform(test_evaluation_config_mobilenetv3_large)

    # Create CustomImageDataset and DataLoader for the test set
    test_dataset_mobilenetv3_large = CustomImageDataset(
        clean_test_df,
        transform=transform_test_mobilenetv3_large,
        labels_map=test_evaluation_config_mobilenetv3_large['labels_map']
    )

    test_loader_mobilenetv3_large = DataLoader(
        test_dataset_mobilenetv3_large,
        batch_size=test_evaluation_config_mobilenetv3_large['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # Evaluate the model on the test set
    test_results_mobilenetv3_large = evaluate_model(
        model=final_model_mobilenetv3_large,
        dataloader=test_loader_mobilenetv3_large,
        device=device,
        class_names=test_evaluation_config_mobilenetv3_large['class_names'],
        title="MobileNetV3 Large Final Test Set Evaluation",
        show_plots=True # Set to False if you don't want plots immediately
    )

    # Print the cumulative accuracy
    cumulative_accuracy_mobilenetv3_large = test_results_mobilenetv3_large['accuracy']
    print(f"\nCumulative Accuracy of MobileNetV3 Large on the test set: {cumulative_accuracy_mobilenetv3_large:.4f}")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path_to_load_mobilenetv3_large}. Please ensure the training completed successfully and the file exists.")
except Exception as e:
    print(f"An error occurred while loading or evaluating the MobileNetV3 Large model: {e}")

# %% [markdown] Cell 71
# save the model

# %% Cell 72
import json
import os
import numpy as np

# Define the output file path for the MobileNetV3 Large test results
output_file_path_mobilenetv3_large = os.path.join(
    gdrive_base_output_dir,
    'mobilenetv3_large_100_lung_clean_cumulative_test_results.json'
)

# Prepare the dictionary for JSON serialization by converting numpy arrays to lists
serializable_test_results_mobilenetv3_large = {
    k: (v.tolist() if isinstance(v, np.ndarray) else v)
    for k, v in test_results_mobilenetv3_large.items()
}

# Save the test results to a JSON file
try:
    with open(output_file_path_mobilenetv3_large, 'w') as f:
        json.dump(serializable_test_results_mobilenetv3_large, f, indent=4)
    print(f"MobileNetV3 Large test results saved to: {output_file_path_mobilenetv3_large}")
except Exception as e:
    print(f"An error occurred while saving MobileNetV3 Large test results: {e}")

# %% [markdown] Cell 73
# Ensamble Accuracy

# %% Cell 74
import torch
import timm
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

set_all_seeds(SEED)

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Model details for MobileNetV3 Large
model_name_mobilenetv3_large = 'mobilenetv3_large_100_lung_clean'
model_to_load_architecture_mobilenetv3_large = 'mobilenetv3_large_100'

# Dictionary to store all loaded models and their configs
loaded_mobilenetv3_large_fold_models = {}

print(f"Loading all {COMMON_CONFIG['num_folds']} folds for {model_name_mobilenetv3_large}...")

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    model_save_dir = os.path.join(gdrive_base_output_dir, model_name_mobilenetv3_large)

    fold_weight_path = os.path.join(
        model_save_dir,
        f"{model_name_mobilenetv3_large}_fold_{fold_number}.pt"
    )

    fold_config_path = os.path.join(
        model_save_dir,
        f"{model_name_mobilenetv3_large}_fold_{fold_number}_config.json"
    )

    try:
        with open(fold_config_path, "r") as f:
            fold_config = json.load(f)

        # Convert mean/std back to numpy arrays
        fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
        fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

        model_mobilenetv3_large = timm.create_model(
            model_to_load_architecture_mobilenetv3_large,
            pretrained=False, # We are loading custom trained weights
            num_classes=fold_config['num_classes']
        )

        model_mobilenetv3_large.load_state_dict(
            torch.load(fold_weight_path, map_location=device)
        )

        model_mobilenetv3_large = model_mobilenetv3_large.to(device)
        model_mobilenetv3_large.eval() # Set model to evaluation mode

        loaded_mobilenetv3_large_fold_models[f'fold_{fold_number}'] = {
            'model': model_mobilenetv3_large,
            'config': fold_config
        }

        print(f"MobileNetV3 Large Fold {fold_number} loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Weights or config not found for Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
    except Exception as e:
        print(f"An error occurred while loading MobileNetV3 Large Fold {fold_number}: {e}")

print(f"\nAll {len(loaded_mobilenetv3_large_fold_models)} MobileNetV3 Large models loaded.")

# Prepare for ensemble prediction
all_fold_probs_mobilenetv3_large = []
y_true_final_mobilenetv3_large = None
class_names_mobilenetv3_large = None

if loaded_mobilenetv3_large_fold_models:
    print(f"\nEvaluating all {COMMON_CONFIG['num_folds']} MobileNetV3 Large fold models on the final test set for ensemble...")

    for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
        print(f"\n--- Collecting predictions for MobileNetV3 Large Fold {fold_number} ---")

        model_data = loaded_mobilenetv3_large_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            print(f"Skipping Fold {fold_number} due to missing model data.")
            continue

        model = model_data['model']
        fold_config = model_data['config']

        # Build test transformations using the fold-specific mean/std
        transform_test = build_val_test_transform(fold_config)

        # Create CustomImageDataset and DataLoader for the test set
        test_dataset = CustomImageDataset(
            clean_test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"MobileNetV3 Large Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        all_fold_probs_mobilenetv3_large.append(fold_probs)

        if y_true_final_mobilenetv3_large is None:
            y_true_final_mobilenetv3_large = fold_labels
            class_names_mobilenetv3_large = fold_config['class_names']
        else:
            assert np.array_equal(y_true_final_mobilenetv3_large, fold_labels), "Label order changed between folds!"

    if all_fold_probs_mobilenetv3_large:
        # Average the probabilities across all folds
        avg_probs_mobilenetv3_large = np.mean(np.stack(all_fold_probs_mobilenetv3_large, axis=0), axis=0)
        y_pred_final_mobilenetv3_large = np.argmax(avg_probs_mobilenetv3_large, axis=1)

        # Calculate ensemble metrics
        acc_ensemble_mobilenetv3_large = accuracy_score(y_true_final_mobilenetv3_large, y_pred_final_mobilenetv3_large)
        prec_ensemble_mobilenetv3_large = precision_score(y_true_final_mobilenetv3_large, y_pred_final_mobilenetv3_large, average="weighted", zero_division=0)
        rec_ensemble_mobilenetv3_large = recall_score(y_true_final_mobilenetv3_large, y_pred_final_mobilenetv3_large, average="weighted", zero_division=0)
        f1_ensemble_mobilenetv3_large = f1_score(y_true_final_mobilenetv3_large, y_pred_final_mobilenetv3_large, average="weighted", zero_division=0)

        print("\n========== FINAL CLEAN TEST RESULT: MobileNetV3 Large 5-Fold Ensemble ==========")
        print(f"Accuracy : {acc_ensemble_mobilenetv3_large:.4f}")
        print(f"Precision: {prec_ensemble_mobilenetv3_large:.4f}")
        print(f"Recall   : {rec_ensemble_mobilenetv3_large:.4f}")
        print(f"F1-score : {f1_ensemble_mobilenetv3_large:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_true_final_mobilenetv3_large,
            y_pred_final_mobilenetv3_large,
            target_names=class_names_mobilenetv3_large,
            zero_division=0
        ))

        cm_ensemble_mobilenetv3_large = confusion_matrix(
            y_true_final_mobilenetv3_large,
            y_pred_final_mobilenetv3_large,
            labels=range(len(class_names_mobilenetv3_large))
        )

        print("\nConfusion Matrix:")
        print(cm_ensemble_mobilenetv3_large)
    else:
        print("No MobileNetV3 Large fold probabilities were collected.")
else:
    print("No MobileNetV3 Large models were loaded to perform ensemble evaluation.")

# %% [markdown] Cell 75
# tf_efficienetv2

# %% Cell 76
import os

# Define the model specifications for EfficientNetV2-S
MODEL_SPECS_EFFICIENTNETV2_S = [
    {
        'model_to_load': 'tf_efficientnetv2_s',
        'model_name': 'efficientnetv2_s_lung_clean',
    }
]

# Ensure gdrive_base_output_dir exists (already defined in previous cells)
os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

# Dictionary to store training histories for all models
all_model_histories_efficientnetv2_s = {}

# Loop through each model specification and train the model using K-fold cross-validation
for model_spec in MODEL_SPECS_EFFICIENTNETV2_S:
    model_name = model_spec['model_name']
    print(f"\n{'='*50}")
    print(f"Initiating K-fold training for {model_name}")
    print(f"{'-'*50}")

    # Call the train_single_model_same_config function
    # folds_info is available from previous cells and contains the common config for all folds
    histories = train_single_model_same_config(
        model_spec=model_spec,
        folds_info=folds_info,
        base_output_dir=gdrive_base_output_dir
    )

    all_model_histories_efficientnetv2_s[model_name] = histories

print("\n" + "="*50)
print("EfficientNetV2-S model processed. 'all_model_histories_efficientnetv2_s' dictionary contains training histories.")
print(f"Keys in all_model_histories_efficientnetv2_s: {list(all_model_histories_efficientnetv2_s.keys())}")
print("="*50)

# %% [markdown] Cell 77
# Load the model

# %% Cell 78
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# The `plot_training_curves` function is defined in a previous cell.
# Ensuring it's available in this scope for clarity if this cell is run independently.
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    # Adjust subplot creation for single or multiple folds
    if num_folds == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = [axes] # Make it iterable for consistent loop
    else:
        fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    fig.suptitle(f'Training and Validation Curves for {model_name}', fontsize=16)

    for i, history in enumerate(histories):
        # Plot Loss
        ax_loss = axes[i][0] if num_folds > 1 else axes[0]
        ax_loss.plot(history['train_loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'Fold {i+1} - Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        # Plot Accuracy
        ax_acc = axes[i][1] if num_folds > 1 else axes[1]
        ax_acc.plot(history['train_acc'], label='Train Accuracy')
        ax_acc.plot(history['val_acc'], label='Validation Accuracy')
        ax_acc.set_title(f'Fold {i+1} - Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Define the path to the saved history file for EfficientNetV2-S
# gdrive_base_output_dir is assumed to be defined from previous cells
model_name_efficientnetv2_s = 'efficientnetv2_s_lung_clean'
load_history_path_efficientnetv2_s = os.path.join(gdrive_base_output_dir, model_name_efficientnetv2_s + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_efficientnetv2_s, 'r') as f:
        loaded_histories_efficientnetv2_s = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_efficientnetv2_s}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(loaded_histories_efficientnetv2_s)}")
    for i, history in enumerate(loaded_histories_efficientnetv2_s):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(loaded_histories_efficientnetv2_s, model_name_efficientnetv2_s)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_efficientnetv2_s}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_efficientnetv2_s}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown] Cell 79
# Ensemble Accuracy

# %% Cell 80
import torch
import timm
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

set_all_seeds(SEED)

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Model details for EfficientNetV2-S
model_name_efficientnetv2_s = 'efficientnetv2_s_lung_clean'
model_to_load_architecture_efficientnetv2_s = 'tf_efficientnetv2_s'

# Dictionary to store all loaded models and their configs
loaded_efficientnetv2_s_fold_models = {}

print(f"Loading all {COMMON_CONFIG['num_folds']} folds for {model_name_efficientnetv2_s}...")

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    model_save_dir = os.path.join(gdrive_base_output_dir, model_name_efficientnetv2_s)

    fold_weight_path = os.path.join(
        model_save_dir,
        f"{model_name_efficientnetv2_s}_fold_{fold_number}.pt"
    )

    fold_config_path = os.path.join(
        model_save_dir,
        f"{model_name_efficientnetv2_s}_fold_{fold_number}_config.json"
    )

    try:
        with open(fold_config_path, "r") as f:
            fold_config = json.load(f)

        # Convert mean/std back to numpy arrays
        fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
        fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

        model_efficientnetv2_s = timm.create_model(
            model_to_load_architecture_efficientnetv2_s,
            pretrained=False, # We are loading custom trained weights
            num_classes=fold_config['num_classes']
        )

        model_efficientnetv2_s.load_state_dict(
            torch.load(fold_weight_path, map_location=device)
        )

        model_efficientnetv2_s = model_efficientnetv2_s.to(device)
        model_efficientnetv2_s.eval() # Set model to evaluation mode

        loaded_efficientnetv2_s_fold_models[f'fold_{fold_number}'] = {
            'model': model_efficientnetv2_s,
            'config': fold_config
        }

        print(f"EfficientNetV2-S Fold {fold_number} loaded successfully.")

    except FileNotFoundError:
        print(f"Error: Weights or config not found for Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
    except Exception as e:
        print(f"An error occurred while loading EfficientNetV2-S Fold {fold_number}: {e}")

print(f"\nAll {len(loaded_efficientnetv2_s_fold_models)} EfficientNetV2-S models loaded.")

# Prepare for ensemble prediction
all_fold_probs_efficientnetv2_s = []
y_true_final_efficientnetv2_s = None
class_names_efficientnetv2_s = None

if loaded_efficientnetv2_s_fold_models:
    print(f"\nEvaluating all {COMMON_CONFIG['num_folds']} EfficientNetV2-S fold models on the final test set for ensemble...")

    for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
        print(f"\n--- Collecting predictions for EfficientNetV2-S Fold {fold_number} ---")

        model_data = loaded_efficientnetv2_s_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            print(f"Skipping Fold {fold_number} due to missing model data.")
            continue

        model = model_data['model']
        fold_config = model_data['config']

        # Build test transformations using the fold-specific mean/std
        transform_test = build_val_test_transform(fold_config)

        # Create CustomImageDataset and DataLoader for the test set
        test_dataset = CustomImageDataset(
            clean_test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"EfficientNetV2-S Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        all_fold_probs_efficientnetv2_s.append(fold_probs)

        if y_true_final_efficientnetv2_s is None:
            y_true_final_efficientnetv2_s = fold_labels
            class_names_efficientnetv2_s = fold_config['class_names']
        else:
            assert np.array_equal(y_true_final_efficientnetv2_s, fold_labels), "Label order changed between folds!"

    if all_fold_probs_efficientnetv2_s:
        # Average the probabilities across all folds
        avg_probs_efficientnetv2_s = np.mean(np.stack(all_fold_probs_efficientnetv2_s, axis=0), axis=0)
        y_pred_final_efficientnetv2_s = np.argmax(avg_probs_efficientnetv2_s, axis=1)

        # Calculate ensemble metrics
        acc_ensemble = accuracy_score(y_true_final_efficientnetv2_s, y_pred_final_efficientnetv2_s)
        prec_ensemble = precision_score(y_true_final_efficientnetv2_s, y_pred_final_efficientnetv2_s, average="weighted", zero_division=0)
        rec_ensemble = recall_score(y_true_final_efficientnetv2_s, y_pred_final_efficientnetv2_s, average="weighted", zero_division=0)
        f1_ensemble = f1_score(y_true_final_efficientnetv2_s, y_pred_final_efficientnetv2_s, average="weighted", zero_division=0)

        print("\n========== FINAL CLEAN TEST RESULT: EfficientNetV2-S 5-Fold Ensemble ==========")
        print(f"Accuracy : {acc_ensemble:.4f}")
        print(f"Precision: {prec_ensemble:.4f}")
        print(f"Recall   : {rec_ensemble:.4f}")
        print(f"F1-score : {f1_ensemble:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_true_final_efficientnetv2_s,
            y_pred_final_efficientnetv2_s,
            target_names=class_names_efficientnetv2_s,
            zero_division=0
        ))

        cm_ensemble = confusion_matrix(
            y_true_final_efficientnetv2_s,
            y_pred_final_efficientnetv2_s,
            labels=range(len(class_names_efficientnetv2_s))
        )

        print("\nConfusion Matrix:")
        print(cm_ensemble)
    else:
        print("No EfficientNetV2-S fold probabilities were collected.")
else:
    print("No EfficientNetV2-S models were loaded to perform ensemble evaluation.")

# %% [markdown] Cell 81
# calculate AUC all models

# %% Cell 82
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

plt.figure(figsize=(12, 9))

# Iterate through each model to calculate and plot its Macro-Average ROC curve
for model_name, (true_var, prob_var) in models_data.items():
    # Handle the naming convention fallback for ground truth labels
    if true_var not in globals():
        alt_true_var = true_var.replace("_final", "")
        if alt_true_var in globals():
            true_var = alt_true_var

    if true_var in globals() and prob_var in globals():
        y_true = globals()[true_var]
        y_probs = globals()[prob_var]

        if y_true is not None and y_probs is not None:
            # Binarize labels for multi-class ROC
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

            # Compute Macro-average ROC curve
            all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, i], y_probs[:, i])[0] for i in range(len(class_names))]))
            mean_tpr = np.zeros_like(all_fpr)

            for i in range(len(class_names)):
                fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            mean_tpr /= len(class_names)

            macro_auc = auc(all_fpr, mean_tpr)

            plt.plot(all_fpr, mean_tpr, lw=2, label=f'{model_name} (AUC = {macro_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Macro-Average ROC Curves for Ensemble Models', fontsize=16)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown] Cell 83
# Feature fusion

# %% Cell 84
import torch
import torch.nn as nn
import timm

class FeatureFusionModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(FeatureFusionModel, self).__init__()

        # Load MobileNetV3 Large and DenseNet121 as feature extractors (num_classes=0 removes the final classification layer)
        print("Loading feature extractors...")
        self.model_mobilenet = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=0)
        self.model_densenet = timm.create_model('densenet121', pretrained=pretrained, num_classes=0)

        # Determine the combined feature dimension dynamically
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_mobile = self.model_mobilenet(dummy_input)
            out_dense = self.model_densenet(dummy_input)

        combined_features_dim = out_mobile.shape[1] + out_dense.shape[1]
        print(f"MobileNetV3 feature dim: {out_mobile.shape[1]}")
        print(f"DenseNet121 feature dim: {out_dense.shape[1]}")
        print(f"Combined feature dim: {combined_features_dim}")

        # Define the final classification layer
        self.classifier = nn.Linear(combined_features_dim, num_classes)

    def forward(self, x):
        # Extract features
        features_mobile = self.model_mobilenet(x)
        features_dense = self.model_densenet(x)

        # Concatenate features along the channel dimension
        fused_features = torch.cat((features_mobile, features_dense), dim=1)

        # Pass through the classifier
        out = self.classifier(fused_features)
        return out

# Instantiate the fusion model
print("\nInitializing Fusion Model...")
fusion_model = FeatureFusionModel(num_classes=num_classes, pretrained=True)
fusion_model = fusion_model.to(device)

print("\nFusion Model Ready!")
# print(fusion_model) # Uncomment to see the full architecture

# %% Cell 85
import os
import json
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Included the missing CustomImageDataset class definition
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, labels_map=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels_map = labels_map

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')

        label_name = self.dataframe.iloc[idx]['label']
        label = self.labels_map[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_fusion_model_kfold(folds_info, base_output_dir, num_epochs=30):
    model_name = "fusion_mobilenetv3_densenet121_clean"
    model_save_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    histories = []

    print(f"\n{'='*60}")
    print(f"Starting K-fold training for Fusion Model: {model_name}")
    print(f"{'='*60}")

    for fold_info in folds_info:
        fold = fold_info['fold']
        train_fold_df = fold_info['train_df']
        val_fold_df = fold_info['val_df']

        fold_config = fold_info['config'].copy()
        fold_config['n_epochs'] = num_epochs # Override epochs if needed for quicker testing
        fold_config['model_name'] = model_name

        print(f"\nStarting Fold {fold}/{fold_config['num_folds']} for {model_name}")

        # Prepare Transforms & Datasets
        transform_train = build_train_transform(fold_config)
        transform_val = build_val_test_transform(fold_config)

        train_dataset = CustomImageDataset(train_fold_df, transform=transform_train, labels_map=fold_config['labels_map'])
        val_dataset = CustomImageDataset(val_fold_df, transform=transform_val, labels_map=fold_config['labels_map'])

        train_loader = DataLoader(train_dataset, batch_size=fold_config['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=fold_config['batch_size'], shuffle=False, num_workers=2)

        # Initialize fresh model for the current fold
        model = FeatureFusionModel(num_classes=fold_config['num_classes'], pretrained=True)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=fold_config.get('label_smoothing', 0.0))
        optimizer = optim.Adam(model.parameters(), lr=fold_config['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=fold_config['epochs_patience'] // 2)

        fold_save_path = os.path.join(model_save_dir, f"{model_name}_fold_{fold}.pt")

        # Train using existing training loop
        model_trained, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=fold_config,
            device=device,
            save_path=fold_save_path
        )

        histories.append(history)

        # Evaluate Validation Set
        print(f"\nValidation evaluation for {model_name} Fold {fold}")
        evaluate_model(
            model_trained,
            val_loader,
            device,
            fold_config['class_names'],
            title=f"{model_name} Fold {fold} Validation",
            show_plots=True
        )

        # Save fold config
        fold_config_to_save = make_json_serializable_config(fold_config)
        with open(os.path.join(model_save_dir, f"{model_name}_fold_{fold}_config.json"), "w") as f:
            json.dump(fold_config_to_save, f, indent=4)

    # Save all histories
    history_save_path = os.path.join(base_output_dir, f"{model_name}_histories_clean.json")
    with open(history_save_path, "w") as f:
        json.dump(histories, f, indent=4)
    print(f"\nSaved history: {history_save_path}")

    # Plot Training Curves
    plot_training_curves(histories, model_name)

    return histories

# Execute the K-fold training for the fusion model
# Using COMMON_CONFIG['n_epochs'] (or change it to a smaller number like 10 for a quicker test)
fusion_histories = train_fusion_model_kfold(
    folds_info=folds_info,
    base_output_dir=gdrive_base_output_dir,
    num_epochs=COMMON_CONFIG['n_epochs']
)

# %% [markdown] Cell 86
# Evalute the classification of feature fusion

# %% Cell 87
import torch
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

set_all_seeds(SEED)

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

model_name_fusion = 'fusion_mobilenetv3_densenet121_clean'

print(f"Loading all {COMMON_CONFIG['num_folds']} folds for {model_name_fusion}...")

loaded_fusion_fold_models = {}

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    model_save_dir = os.path.join(gdrive_base_output_dir, model_name_fusion)

    fold_weight_path = os.path.join(
        model_save_dir,
        f"{model_name_fusion}_fold_{fold_number}.pt"
    )

    fold_config_path = os.path.join(
        model_save_dir,
        f"{model_name_fusion}_fold_{fold_number}_config.json"
    )

    try:
        with open(fold_config_path, "r") as f:
            fold_config = json.load(f)

        # Convert mean/std back to numpy arrays
        fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
        fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

        # Initialize fusion model
        model_fusion = FeatureFusionModel(num_classes=fold_config['num_classes'], pretrained=False)
        model_fusion.load_state_dict(torch.load(fold_weight_path, map_location=device))
        model_fusion = model_fusion.to(device)
        model_fusion.eval()

        loaded_fusion_fold_models[f'fold_{fold_number}'] = {
            'model': model_fusion,
            'config': fold_config
        }

        print(f"Fusion Model Fold {fold_number} loaded successfully.")
    except Exception as e:
        print(f"Error loading Fusion Model Fold {fold_number}: {e}")

print(f"\nAll {len(loaded_fusion_fold_models)} Fusion models loaded.")

# Prepare for ensemble prediction
all_fold_probs_fusion = []
y_true_final_fusion = None
class_names_fusion = None

if loaded_fusion_fold_models:
    print(f"\nEvaluating all {COMMON_CONFIG['num_folds']} Fusion fold models on the final test set for ensemble...")

    for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
        print(f"\n--- Collecting predictions for Fusion Fold {fold_number} ---")

        model_data = loaded_fusion_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            continue

        model = model_data['model']
        fold_config = model_data['config']

        # Build test transformations using the fold-specific mean/std
        transform_test = build_val_test_transform(fold_config)

        # Create CustomImageDataset and DataLoader for the test set
        test_dataset = CustomImageDataset(
            clean_test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"Fusion Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        all_fold_probs_fusion.append(fold_probs)

        if y_true_final_fusion is None:
            y_true_final_fusion = fold_labels
            class_names_fusion = fold_config['class_names']

    if all_fold_probs_fusion:
        # Average the probabilities across all folds
        avg_probs_fusion = np.mean(np.stack(all_fold_probs_fusion, axis=0), axis=0)
        y_pred_final_fusion = np.argmax(avg_probs_fusion, axis=1)

        # Calculate ensemble metrics
        acc_ensemble = accuracy_score(y_true_final_fusion, y_pred_final_fusion)
        prec_ensemble = precision_score(y_true_final_fusion, y_pred_final_fusion, average="weighted", zero_division=0)
        rec_ensemble = recall_score(y_true_final_fusion, y_pred_final_fusion, average="weighted", zero_division=0)
        f1_ensemble = f1_score(y_true_final_fusion, y_pred_final_fusion, average="weighted", zero_division=0)

        print("\n========== FINAL CLEAN TEST RESULT: Fusion Model 5-Fold Ensemble ==========")
        print(f"Accuracy : {acc_ensemble:.4f}")
        print(f"Precision: {prec_ensemble:.4f}")
        print(f"Recall   : {rec_ensemble:.4f}")
        print(f"F1-score : {f1_ensemble:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true_final_fusion, y_pred_final_fusion, target_names=class_names_fusion, zero_division=0))

        cm_ensemble = confusion_matrix(y_true_final_fusion, y_pred_final_fusion, labels=range(len(class_names_fusion)))

        print("\nConfusion Matrix:")
        print(cm_ensemble)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_fusion, yticklabels=class_names_fusion)
        plt.title("Fusion Model 5-Fold Ensemble Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

# %% [markdown] Cell 88
# Evalute the Unseen test data

# %% Cell 89
from sklearn.metrics import roc_auc_score
import json
import os

# Calculate Macro AUC for the Fusion Model
try:
    macro_auc_fusion = roc_auc_score(y_true_final_fusion, avg_probs_fusion, multi_class='ovr', average='macro')
    print(f"Fusion Model Macro AUC: {macro_auc_fusion:.4f}\n")

    # Print per-class AUC
    print("Per-class AUC:")
    for i, class_name in enumerate(class_names_fusion):
        class_auc = roc_auc_score(y_true_final_fusion == i, avg_probs_fusion[:, i])
        print(f"  - {class_name:<31} : {class_auc:.4f}")
except Exception as e:
    macro_auc_fusion = None
    print(f"Could not calculate AUC: {e}")

# Save the ensemble results for the fusion model
fusion_ensemble_output_file = os.path.join(
    gdrive_base_output_dir,
    'fusion_mobilenetv3_densenet121_clean_ensemble_test_results.json'
)

serializable_results_fusion = {
    'accuracy': acc_ensemble,
    'precision': prec_ensemble,
    'recall': rec_ensemble,
    'f1_score': f1_ensemble,
    'macro_auc': macro_auc_fusion,
    'confusion_matrix': cm_ensemble.tolist(),
    'class_names': class_names_fusion
}

try:
    with open(fusion_ensemble_output_file, 'w') as f:
        json.dump(serializable_results_fusion, f, indent=4)
    print(f"\nFusion Model 5-fold ensemble results successfully saved to:\n{fusion_ensemble_output_file}")
except Exception as e:
    print(f"An error occurred while saving Fusion Model ensemble results: {e}")

# %% [markdown] Cell 90
# Evaluate the avverage accuracy

# %% Cell 91
import numpy as np

if 'fusion_histories' in locals():
    fold_best_accuracies = []
    for i, history in enumerate(fusion_histories):
        # Find the maximum validation accuracy achieved in each fold
        best_val_acc = np.max(history['val_acc'])
        fold_best_accuracies.append(best_val_acc)
        print(f"Fold {i+1} Best Validation Accuracy: {best_val_acc:.4f}")

    if fold_best_accuracies:
        avg_accuracy = np.mean(fold_best_accuracies)
        print(f"\nAverage Validation Accuracy across all Feature Fusion folds: {avg_accuracy:.4f}")
    else:
        print("No validation accuracies found for Feature Fusion folds.")
else:
    print("Error: 'fusion_histories' not found in memory. Please ensure the Feature Fusion model training was completed.")

# %% Cell 92
import json
import os

# Ensure the base directory and model name are defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
model_name_fusion = 'fusion_mobilenetv3_densenet121_clean'
load_history_path_fusion = os.path.join(gdrive_base_output_dir, model_name_fusion + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_fusion, 'r') as f:
        fusion_histories = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_fusion}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(fusion_histories)}")
    for i, history in enumerate(fusion_histories):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(fusion_histories, model_name_fusion)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_fusion}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_fusion}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown] Cell 93
# Uncertainty Analysis

# %% Cell 94
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.calibration import calibration_curve

# 1. Load variables from the MobileNetV3 + DenseNet121 Fusion model
y_true = y_true_final_fusion
probs = avg_probs_fusion
y_pred = np.argmax(probs, axis=1)
confidences = np.max(probs, axis=1)
classes = class_names_fusion

# 2. Calculate entropy (uncertainty) in bits
uncertainties = entropy(probs, axis=1, base=2)

correct_mask = (y_true == y_pred)
incorrect_mask = ~correct_mask

# Setup matplotlib figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Comprehensive Uncertainty Analysis\n(MobileNetV3 + DenseNet121 Fusion Ensemble)', fontsize=18)

# --- Plot 1: Entropy Distribution ---
sns.histplot(uncertainties[correct_mask], color='blue', alpha=0.5, label='Correct', stat='density', bins=20, kde=True, ax=axes[0, 0])
if np.sum(incorrect_mask) > 0:
    sns.histplot(uncertainties[incorrect_mask], color='red', alpha=0.5, label='Incorrect', stat='density', bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Predictive Entropy Distribution')
axes[0, 0].set_xlabel('Entropy (bits)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# --- Plot 2: Confidence Distribution ---
sns.histplot(confidences[correct_mask], color='green', alpha=0.5, label='Correct', stat='density', bins=20, kde=True, ax=axes[0, 1])
if np.sum(incorrect_mask) > 0:
    sns.histplot(confidences[incorrect_mask], color='orange', alpha=0.5, label='Incorrect', stat='density', bins=20, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Confidence (Max Probability) Distribution')
axes[0, 1].set_xlabel('Confidence Score')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# --- Plot 3: Reliability Diagram (Calibration Curve) ---
# Convert to binary classification for calibration curve (Is the prediction correct?)
y_true_bin = (y_true == y_pred).astype(int)
prob_true, prob_pred = calibration_curve(y_true_bin, confidences, n_bins=10, strategy='uniform')
axes[1, 0].plot(prob_pred, prob_true, marker='o', color='purple', label='Model Calibration')
axes[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
axes[1, 0].set_title('Reliability Diagram (Calibration)')
axes[1, 0].set_xlabel('Mean Predicted Confidence')
axes[1, 0].set_ylabel('Fraction of Positives (Accuracy)')
axes[1, 0].legend()

# --- Plot 4: Average Uncertainty Confusion Matrix ---
num_classes = len(classes)
avg_uncert_matrix = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        mask = (y_true == i) & (y_pred == j)
        if np.sum(mask) > 0:
            avg_uncert_matrix[i, j] = np.mean(uncertainties[mask])
        else:
            avg_uncert_matrix[i, j] = np.nan # Use NaN to represent no predictions in this cell

sns.heatmap(
    avg_uncert_matrix,
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    xticklabels=classes,
    yticklabels=classes,
    cbar_kws={'label': 'Average Entropy (bits)'},
    linewidths=1,
    linecolor='lightgray',
    ax=axes[1, 1]
)
axes[1, 1].set_title('Average Predictive Uncertainty by Confusion Matrix Cell')
axes[1, 1].set_xlabel('Predicted Class')
axes[1, 1].set_ylabel('True Class')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Print summary statistics
print("--- Uncertainty Analysis Summary ---")
print(f"Mean Entropy (Correct)      : {np.mean(uncertainties[correct_mask]):.4f} bits")
if np.sum(incorrect_mask) > 0:
    print(f"Mean Entropy (Incorrect)    : {np.mean(uncertainties[incorrect_mask]):.4f} bits")
print(f"Mean Confidence (Correct)   : {np.mean(confidences[correct_mask]):.4f}")
if np.sum(incorrect_mask) > 0:
    print(f"Mean Confidence (Incorrect) : {np.mean(confidences[incorrect_mask]):.4f}")

# %% [markdown] Cell 95
# feature  fusion between MobileNetV2 and Densenet201

# %% [markdown] Cell 96
# Define Fusion

# %% Cell 97
import torch
import torch.nn as nn
import timm

class FeatureFusionMobileNetV2DenseNet201(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(FeatureFusionMobileNetV2DenseNet201, self).__init__()

        # Load MobileNetV2 and DenseNet201 as feature extractors (num_classes=0 removes the final classification layer)
        print("Loading feature extractors...")
        self.model_mobilenet = timm.create_model('mobilenetv2_100', pretrained=pretrained, num_classes=0)
        self.model_densenet = timm.create_model('densenet201', pretrained=pretrained, num_classes=0)

        # Determine the combined feature dimension dynamically
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_mobile = self.model_mobilenet(dummy_input)
            out_dense = self.model_densenet(dummy_input)

        combined_features_dim = out_mobile.shape[1] + out_dense.shape[1]
        print(f"MobileNetV2 feature dim: {out_mobile.shape[1]}")
        print(f"DenseNet201 feature dim: {out_dense.shape[1]}")
        print(f"Combined feature dim: {combined_features_dim}")

        # Define the final classification layer
        self.classifier = nn.Linear(combined_features_dim, num_classes)

    def forward(self, x):
        # Extract features
        features_mobile = self.model_mobilenet(x)
        features_dense = self.model_densenet(x)

        # Concatenate features along the channel dimension
        fused_features = torch.cat((features_mobile, features_dense), dim=1)

        # Pass through the classifier
        out = self.classifier(fused_features)
        return out

# Instantiate the fusion model
print("\nInitializing MobileNetV2 + DenseNet201 Fusion Model...")
fusion_model_v2_d201 = FeatureFusionMobileNetV2DenseNet201(num_classes=num_classes, pretrained=True)
fusion_model_v2_d201 = fusion_model_v2_d201.to(device)

print("\nFusion Model Ready!")
# print(fusion_model_v2_d201) # Uncomment to see the full architecture

# %% Cell 98
import os
import json
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

def train_fusion_v2_d201_kfold(folds_info, base_output_dir, num_epochs=30):
    model_name = "fusion_mobilenetv2_densenet201_clean"
    model_save_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    histories = []

    print(f"\n{'='*60}")
    print(f"Starting K-fold training for Fusion Model: {model_name}")
    print(f"{'='*60}")

    for fold_info in folds_info:
        fold = fold_info['fold']
        train_fold_df = fold_info['train_df']
        val_fold_df = fold_info['val_df']

        fold_config = fold_info['config'].copy()
        fold_config['n_epochs'] = num_epochs
        fold_config['model_name'] = model_name

        print(f"\nStarting Fold {fold}/{fold_config['num_folds']} for {model_name}")

        # Prepare Transforms & Datasets
        transform_train = build_train_transform(fold_config)
        transform_val = build_val_test_transform(fold_config)

        train_dataset = CustomImageDataset(train_fold_df, transform=transform_train, labels_map=fold_config['labels_map'])
        val_dataset = CustomImageDataset(val_fold_df, transform=transform_val, labels_map=fold_config['labels_map'])

        train_loader = DataLoader(train_dataset, batch_size=fold_config['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=fold_config['batch_size'], shuffle=False, num_workers=2)

        # Initialize fresh model for the current fold
        model = FeatureFusionMobileNetV2DenseNet201(num_classes=fold_config['num_classes'], pretrained=True)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=fold_config.get('label_smoothing', 0.0))
        optimizer = optim.Adam(model.parameters(), lr=fold_config['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=fold_config['epochs_patience'] // 2)

        fold_save_path = os.path.join(model_save_dir, f"{model_name}_fold_{fold}.pt")

        # Train using existing training loop
        model_trained, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=fold_config,
            device=device,
            save_path=fold_save_path
        )

        histories.append(history)

        # Evaluate Validation Set
        print(f"\nValidation evaluation for {model_name} Fold {fold}")
        evaluate_model(
            model_trained,
            val_loader,
            device,
            fold_config['class_names'],
            title=f"{model_name} Fold {fold} Validation",
            show_plots=True
        )

        # Save fold config
        fold_config_to_save = make_json_serializable_config(fold_config)
        with open(os.path.join(model_save_dir, f"{model_name}_fold_{fold}_config.json"), "w") as f:
            json.dump(fold_config_to_save, f, indent=4)

    # Save all histories
    history_save_path = os.path.join(base_output_dir, f"{model_name}_histories_clean.json")
    with open(history_save_path, "w") as f:
        json.dump(histories, f, indent=4)
    print(f"\nSaved history: {history_save_path}")

    # Plot Training Curves
    plot_training_curves(histories, model_name)

    return histories

# Execute the K-fold training for the new fusion model
# Using COMMON_CONFIG['n_epochs'] for the full training process
fusion_v2_d201_histories = train_fusion_v2_d201_kfold(
    folds_info=folds_info,
    base_output_dir=gdrive_base_output_dir,
    num_epochs=COMMON_CONFIG['n_epochs']
)

# %% [markdown] Cell 99
# Evaluate the classification

# %% Cell 100
import torch
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

set_all_seeds(SEED)

# Ensure gdrive_base_output_dir is defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

model_name_fusion_v2_d201 = 'fusion_mobilenetv2_densenet201_clean'

print(f"Loading all {COMMON_CONFIG['num_folds']} folds for {model_name_fusion_v2_d201}...")

loaded_fusion_v2_d201_fold_models = {}

for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
    model_save_dir = os.path.join(gdrive_base_output_dir, model_name_fusion_v2_d201)

    fold_weight_path = os.path.join(
        model_save_dir,
        f"{model_name_fusion_v2_d201}_fold_{fold_number}.pt"
    )

    fold_config_path = os.path.join(
        model_save_dir,
        f"{model_name_fusion_v2_d201}_fold_{fold_number}_config.json"
    )

    try:
        with open(fold_config_path, "r") as f:
            fold_config = json.load(f)

        # Convert mean/std back to numpy arrays
        fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
        fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

        # Initialize fusion model
        model_fusion = FeatureFusionMobileNetV2DenseNet201(num_classes=fold_config['num_classes'], pretrained=False)
        model_fusion.load_state_dict(torch.load(fold_weight_path, map_location=device))
        model_fusion = model_fusion.to(device)
        model_fusion.eval()

        loaded_fusion_v2_d201_fold_models[f'fold_{fold_number}'] = {
            'model': model_fusion,
            'config': fold_config
        }

        print(f"Fusion Model Fold {fold_number} loaded successfully.")
    except Exception as e:
        print(f"Error loading Fusion Model Fold {fold_number}: {e}")

print(f"\nAll {len(loaded_fusion_v2_d201_fold_models)} Fusion models loaded.")

# Prepare for ensemble prediction
all_fold_probs_fusion_v2_d201 = []
y_true_final_fusion_v2_d201 = None
class_names_fusion_v2_d201 = None

if loaded_fusion_v2_d201_fold_models:
    print(f"\nEvaluating all {COMMON_CONFIG['num_folds']} Fusion fold models on the final test set for ensemble...")

    for fold_number in range(1, COMMON_CONFIG['num_folds'] + 1):
        print(f"\n--- Collecting predictions for Fusion Fold {fold_number} ---")

        model_data = loaded_fusion_v2_d201_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            continue

        model = model_data['model']
        fold_config = model_data['config']

        # Build test transformations using the fold-specific mean/std
        transform_test = build_val_test_transform(fold_config)

        # Create CustomImageDataset and DataLoader for the test set
        test_dataset = CustomImageDataset(
            clean_test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=fold_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"Fusion Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        all_fold_probs_fusion_v2_d201.append(fold_probs)

        if y_true_final_fusion_v2_d201 is None:
            y_true_final_fusion_v2_d201 = fold_labels
            class_names_fusion_v2_d201 = fold_config['class_names']

    if all_fold_probs_fusion_v2_d201:
        # Average the probabilities across all folds
        avg_probs_fusion_v2_d201 = np.mean(np.stack(all_fold_probs_fusion_v2_d201, axis=0), axis=0)
        y_pred_final_fusion_v2_d201 = np.argmax(avg_probs_fusion_v2_d201, axis=1)

        # Calculate ensemble metrics
        acc_ensemble = accuracy_score(y_true_final_fusion_v2_d201, y_pred_final_fusion_v2_d201)
        prec_ensemble = precision_score(y_true_final_fusion_v2_d201, y_pred_final_fusion_v2_d201, average="weighted", zero_division=0)
        rec_ensemble = recall_score(y_true_final_fusion_v2_d201, y_pred_final_fusion_v2_d201, average="weighted", zero_division=0)
        f1_ensemble = f1_score(y_true_final_fusion_v2_d201, y_pred_final_fusion_v2_d201, average="weighted", zero_division=0)

        print("\n========== FINAL CLEAN TEST RESULT: MobileNetV2 + DenseNet201 Fusion 5-Fold Ensemble ==========")
        print(f"Accuracy : {acc_ensemble:.4f}")
        print(f"Precision: {prec_ensemble:.4f}")
        print(f"Recall   : {rec_ensemble:.4f}")
        print(f"F1-score : {f1_ensemble:.4f}")

        try:
            macro_auc_fusion = roc_auc_score(y_true_final_fusion_v2_d201, avg_probs_fusion_v2_d201, multi_class='ovr', average='macro')
            print(f"Macro AUC: {macro_auc_fusion:.4f}\n")
        except Exception as e:
            macro_auc_fusion = None
            print(f"Could not calculate AUC: {e}\n")

        print("\nClassification Report:")
        print(classification_report(y_true_final_fusion_v2_d201, y_pred_final_fusion_v2_d201, target_names=class_names_fusion_v2_d201, zero_division=0))

        cm_ensemble = confusion_matrix(y_true_final_fusion_v2_d201, y_pred_final_fusion_v2_d201, labels=range(len(class_names_fusion_v2_d201)))

        print("\nConfusion Matrix:")
        print(cm_ensemble)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_fusion_v2_d201, yticklabels=class_names_fusion_v2_d201)
        plt.title("MobileNetV2 + DenseNet201 Fusion 5-Fold Ensemble Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # Save the ensemble results
        fusion_ensemble_output_file = os.path.join(
            gdrive_base_output_dir,
            'fusion_mobilenetv2_densenet201_clean_ensemble_test_results.json'
        )

        serializable_results_fusion = {
            'accuracy': acc_ensemble,
            'precision': prec_ensemble,
            'recall': rec_ensemble,
            'f1_score': f1_ensemble,
            'macro_auc': macro_auc_fusion,
            'confusion_matrix': cm_ensemble.tolist(),
            'class_names': class_names_fusion_v2_d201
        }

        try:
            with open(fusion_ensemble_output_file, 'w') as f:
                json.dump(serializable_results_fusion, f, indent=4)
            print(f"\nFusion Model 5-fold ensemble results successfully saved to:\n{fusion_ensemble_output_file}")
        except Exception as e:
            print(f"An error occurred while saving Fusion Model ensemble results: {e}")

# %% [markdown] Cell 101
# Evaluate commulative accuarcy

# %% Cell 102
import numpy as np

if 'fusion_v2_d201_histories' in locals():
    fold_best_accuracies = []
    for i, history in enumerate(fusion_v2_d201_histories):
        # Find the maximum validation accuracy achieved in each fold
        best_val_acc = np.max(history['val_acc'])
        fold_best_accuracies.append(best_val_acc)
        print(f"Fold {i+1} Best Validation Accuracy: {best_val_acc:.4f}")

    if fold_best_accuracies:
        avg_accuracy = np.mean(fold_best_accuracies)
        print(f"\nAverage Validation Accuracy across all MobileNetV2 + DenseNet201 Fusion folds: {avg_accuracy:.4f}")
    else:
        print("No validation accuracies found for this fusion model.")
else:
    print("Error: 'fusion_v2_d201_histories' not found in memory. Please ensure the model training was completed.")

# %% Cell 103
import json
import os

# Ensure the base directory and model name are defined
gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'
model_name_fusion_v2_d201 = 'fusion_mobilenetv2_densenet201_clean'
load_history_path_fusion_v2_d201 = os.path.join(gdrive_base_output_dir, model_name_fusion_v2_d201 + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_fusion_v2_d201, 'r') as f:
        fusion_v2_d201_histories_loaded = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_fusion_v2_d201}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(fusion_v2_d201_histories_loaded)}")
    for i, history in enumerate(fusion_v2_d201_histories_loaded):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(fusion_v2_d201_histories_loaded, model_name_fusion_v2_d201)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_fusion_v2_d201}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_fusion_v2_d201}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% [markdown] Cell 104
# Branch-wise Grad-CAM for the best fusion model, especially MobileNetV2 + DenseNet201.

# %% Cell 105
import os
import json
import cv2
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

# ============================================================
# SETTINGS
# ============================================================

BASE_OUTPUT_DIR = "/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE"

MODEL_NAME = "fusion_mobilenetv2_densenet201_clean"
MODEL_TO_LOAD_MOBILE = "mobilenetv2_100"
MODEL_TO_LOAD_DENSE = "densenet201"

# Choose one fold for Grad-CAM
FOLD = 1

# Change this index to visualize another image from the saved final test set
IMAGE_INDEX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_DIR = os.path.join(BASE_OUTPUT_DIR, MODEL_NAME)

weight_path = os.path.join(
    MODEL_DIR,
    f"{MODEL_NAME}_fold_{FOLD}.pt"
)

config_path = os.path.join(
    MODEL_DIR,
    f"{MODEL_NAME}_fold_{FOLD}_config.json"
)

test_csv_path = os.path.join(
    BASE_OUTPUT_DIR,
    "clean_final_test_split.csv"
)

# ============================================================
# LOAD CONFIG AND TEST CSV
# ============================================================

with open(config_path, "r") as f:
    fold_config = json.load(f)

fold_config["mean_overall"] = np.array(fold_config["mean_overall"])
fold_config["std_overall"] = np.array(fold_config["std_overall"])

class_names = fold_config["class_names"]
labels_map = fold_config["labels_map"]
idx_to_class = {v: k for k, v in labels_map.items()}

clean_test_df = pd.read_csv(test_csv_path)

# ============================================================
# TRANSFORM & LOAD IMAGE
# ============================================================

def as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x

test_transform = transforms.Compose([
    transforms.Resize((fold_config['Resize_h'], fold_config['Resize_w'])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=as_list(fold_config['mean_overall']),
        std=as_list(fold_config['std_overall'])
    )
])

img_path = clean_test_df.iloc[IMAGE_INDEX]["filepath"]
true_label_name = clean_test_df.iloc[IMAGE_INDEX]["label"]
true_label_idx = labels_map[true_label_name]

pil_img = Image.open(img_path).convert("RGB")
input_tensor = test_transform(pil_img).unsqueeze(0).to(device)

vis_img = pil_img.resize((fold_config["Resize_w"], fold_config["Resize_h"]))
vis_img_np = np.array(vis_img).astype(np.float32) / 255.0

print("\nImage path:", img_path)
print("True label:", true_label_name)

# ============================================================
# LOAD FUSION MODEL
# ============================================================

class FeatureFusionMobileNetV2DenseNet201(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super(FeatureFusionMobileNetV2DenseNet201, self).__init__()
        self.model_mobilenet = timm.create_model('mobilenetv2_100', pretrained=pretrained, num_classes=0)
        self.model_densenet = timm.create_model('densenet201', pretrained=pretrained, num_classes=0)

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_mobile = self.model_mobilenet(dummy_input)
            out_dense = self.model_densenet(dummy_input)

        combined_features_dim = out_mobile.shape[1] + out_dense.shape[1]
        self.classifier = nn.Linear(combined_features_dim, num_classes)

    def forward(self, x):
        features_mobile = self.model_mobilenet(x)
        features_dense = self.model_densenet(x)
        fused_features = torch.cat((features_mobile, features_dense), dim=1)
        out = self.classifier(fused_features)
        return out

fusion_model = FeatureFusionMobileNetV2DenseNet201(
    num_classes=fold_config["num_classes"],
    pretrained=False
)

state_dict = torch.load(weight_path, map_location=device)
fusion_model.load_state_dict(state_dict)
fusion_model = fusion_model.to(device)
fusion_model.eval()

print("Fusion model loaded successfully.")

# ============================================================
# CUSTOM GRAD-CAM CLASS
# ============================================================

# Updated target layers for more coherent spatial feature extraction
TARGET_LAYER_MOBILE = 'blocks.6'
TARGET_LAYER_DENSE = 'features.norm5'

class _SingleModelGradCAM:
    def __init__(self, model_or_module, target_layer_name):
        self.model_or_module = model_or_module
        self.target_layer = None
        self.activations = None
        self.gradients = None

        for name, module in self.model_or_module.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break

        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer_name}' not found.")

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_cam(self, input_tensor_size):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=(input_tensor_size[2], input_tensor_size[3]), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

class FusionGradCAMGenerator:
    def __init__(self, full_fusion_model, target_layer_mobile, target_layer_dense):
        self.full_fusion_model = full_fusion_model
        self.gradcam_mobile = _SingleModelGradCAM(full_fusion_model.model_mobilenet, target_layer_mobile)
        self.gradcam_dense = _SingleModelGradCAM(full_fusion_model.model_densenet, target_layer_dense)

    def generate(self, input_tensor, class_idx=None):
        self.full_fusion_model.zero_grad(set_to_none=True)

        output = self.full_fusion_model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

        if class_idx is None:
            class_idx = pred_idx

        score = output[0, class_idx]
        score.backward()

        cam_mobile = self.gradcam_mobile.compute_cam(input_tensor.shape)
        cam_dense = self.gradcam_dense.compute_cam(input_tensor.shape)

        combined_cam = (cam_mobile + cam_dense) / 2.0
        combined_cam = combined_cam - combined_cam.min()
        combined_cam = combined_cam / (combined_cam.max() + 1e-8)

        return cam_mobile, cam_dense, combined_cam

    def remove_hooks(self):
        self.gradcam_mobile.remove_hooks()
        self.gradcam_dense.remove_hooks()

# ============================================================
# VISUALIZATION
# ============================================================

num_classes = fold_config["num_classes"]
fig, axes = plt.subplots(num_classes, 4, figsize=(20, 5 * num_classes))
fig.suptitle(f"Branch-wise Grad-CAM: {MODEL_NAME}", fontsize=16, y=1.02)

for i, class_idx in enumerate(range(num_classes)):
    fusion_cam_generator = FusionGradCAMGenerator(fusion_model, TARGET_LAYER_MOBILE, TARGET_LAYER_DENSE)
    cam_mobile, cam_dense, combined_cam = fusion_cam_generator.generate(input_tensor, class_idx=class_idx)
    fusion_cam_generator.remove_hooks()

    class_name = idx_to_class[class_idx]

    axes[i, 0].imshow(vis_img_np)
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Original Image\nClass: {class_name}")

    heatmap_m = cv2.applyColorMap(np.uint8(255 * cam_mobile), cv2.COLORMAP_JET)
    heatmap_m = cv2.cvtColor(heatmap_m, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    axes[i, 1].imshow(np.clip(0.55 * vis_img_np + 0.45 * heatmap_m, 0, 1))
    axes[i, 1].set_title(f"MobileNetV2 CAM")
    axes[i, 1].axis("off")

    heatmap_d = cv2.applyColorMap(np.uint8(255 * cam_dense), cv2.COLORMAP_JET)
    heatmap_d = cv2.cvtColor(heatmap_d, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    axes[i, 2].imshow(np.clip(0.55 * vis_img_np + 0.45 * heatmap_d, 0, 1))
    axes[i, 2].set_title(f"DenseNet201 CAM")
    axes[i, 2].axis("off")

    heatmap_c = cv2.applyColorMap(np.uint8(255 * combined_cam), cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    axes[i, 3].imshow(np.clip(0.55 * vis_img_np + 0.45 * heatmap_c, 0, 1))
    axes[i, 3].set_title(f"Fused CAM")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown] Cell 106
# Train Model without Augmentaion

# %% [markdown] Cell 107
# MobileNetV2 (100)

# %% Cell 108
import os

# Temporarily override build_train_transform to remove all augmentations
# by using the validation/test transform (which only resizes and normalizes).
original_build_train_transform = build_train_transform
build_train_transform = build_val_test_transform

MODEL_SPECS_MOBILENETV2_NO_AUG = [
    {
        'model_to_load': "mobilenetv2_100",
        'model_name': "mobilenetv2_lung_clean_no_aug",
    }
]

os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

all_model_histories_mobilenetv2_no_aug = {}

try:
    for model_spec in MODEL_SPECS_MOBILENETV2_NO_AUG:
        model_name = model_spec['model_name']
        print(f"\n{'='*50}")
        print(f"Initiating K-fold training for {model_name} WITHOUT augmentation")
        print(f"{'-'*50}")

        # Call the train_single_model_same_config function using identical folds_info
        histories = train_single_model_same_config(
            model_spec=model_spec,
            folds_info=folds_info,
            base_output_dir=gdrive_base_output_dir
        )

        all_model_histories_mobilenetv2_no_aug[model_name] = histories

    print("\n" + "="*50)
    print("MobileNetV2 (No Augmentation) model training completed.")
    print("="*50)
finally:
    # Restore the original transform function to avoid affecting subsequent runs
    build_train_transform = original_build_train_transform

# %% Cell 109
import numpy as np

model_name_no_aug = 'mobilenetv2_lung_clean_no_aug'
if 'all_model_histories_mobilenetv2_no_aug' in locals() and model_name_no_aug in all_model_histories_mobilenetv2_no_aug:
    histories_no_aug = all_model_histories_mobilenetv2_no_aug[model_name_no_aug]

    fold_best_accuracies = []
    for i, history in enumerate(histories_no_aug):
        # Find the maximum validation accuracy achieved in each fold
        best_val_acc = np.max(history['val_acc'])
        fold_best_accuracies.append(best_val_acc)
        print(f"Fold {i+1} Best Validation Accuracy: {best_val_acc:.4f}")

    if fold_best_accuracies:
        avg_accuracy = np.mean(fold_best_accuracies)
        print(f"\nAverage Validation Accuracy across all folds (No Augmentation): {avg_accuracy:.4f}")
    else:
        print("No validation accuracies found.")
else:
    print("Error: Training histories not found in memory. Please ensure the model training was completed.")

# %% Cell 110
import torch
import os
import json
import numpy as np
import warnings
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

set_all_seeds(SEED)

def get_ensemble_predictions_for_model_type(
    model_architecture,
    model_name_prefix,
    num_folds,
    base_output_dir_for_model,
    test_df,
    common_config,
    device,
    class_names
):
    print(f"\n--- Loading and collecting predictions for {model_name_prefix} (5-fold ensemble) ---")
    loaded_fold_models = {}

    for fold_number in range(1, num_folds + 1):
        model_save_dir = os.path.join(base_output_dir_for_model, model_name_prefix)
        fold_weight_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}.pt"
        )
        fold_config_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}_config.json"
        )

        try:
            with open(fold_config_path, "r") as f:
                fold_config = json.load(f)

            fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
            fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

            model = build_timm_model(
                model_architecture,
                fold_config
            )
            model.load_state_dict(
                torch.load(fold_weight_path, map_location=device)
            )
            model = model.to(device)
            model.eval()

            loaded_fold_models[f'fold_{fold_number}'] = {
                'model': model,
                'config': fold_config
            }
        except FileNotFoundError:
            print(f"Error: Weights or config not found for {model_name_prefix} Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
            continue
        except Exception as e:
            print(f"An error occurred while loading {model_name_prefix} Fold {fold_number}: {e}")
            continue

    if not loaded_fold_models:
        print(f"No models loaded for {model_name_prefix}. Cannot perform ensemble predictions.")
        return None, None, None

    all_fold_probs = []
    y_true_ret = None

    print(f"Collecting test predictions for {model_name_prefix}...")
    for fold_number in range(1, num_folds + 1):
        model_data = loaded_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            continue

        model = model_data['model']
        fold_config = model_data['config']

        transform_test = build_val_test_transform(fold_config)

        test_dataset = CustomImageDataset(
            test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=common_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"{model_name_prefix} Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)
        all_fold_probs.append(fold_probs)

        if y_true_ret is None:
            y_true_ret = fold_labels
        else:
            if not np.array_equal(y_true_ret, fold_labels):
                warnings.warn(f"Labels for {model_name_prefix} changed between folds, this should not happen with fixed random seed.")

    if not all_fold_probs:
        print(f"No predictions collected for {model_name_prefix}.")
        return None, None, None

    avg_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    return avg_probs, y_true_ret, class_names


model_name_no_aug = 'mobilenetv2_lung_clean_no_aug'
model_architecture_no_aug = 'mobilenetv2_100'

# Get ensemble predictions
avg_probs_no_aug, y_true_no_aug, class_names_no_aug = get_ensemble_predictions_for_model_type(
    model_architecture=model_architecture_no_aug,
    model_name_prefix=model_name_no_aug,
    num_folds=COMMON_CONFIG['num_folds'],
    base_output_dir_for_model=gdrive_base_output_dir,
    test_df=clean_test_df,
    common_config=COMMON_CONFIG,
    device=device,
    class_names=class_names
)

if avg_probs_no_aug is not None:
    y_pred_no_aug = np.argmax(avg_probs_no_aug, axis=1)

    # Calculate metrics
    acc_ensemble_no_aug = accuracy_score(y_true_no_aug, y_pred_no_aug)
    prec_ensemble_no_aug = precision_score(y_true_no_aug, y_pred_no_aug, average="weighted", zero_division=0)
    rec_ensemble_no_aug = recall_score(y_true_no_aug, y_pred_no_aug, average="weighted", zero_division=0)
    f1_ensemble_no_aug = f1_score(y_true_no_aug, y_pred_no_aug, average="weighted", zero_division=0)

    print("\n========== FINAL CLEAN TEST RESULT: MobileNetV2 (No Aug) 5-Fold Ensemble ==========")
    print(f"Accuracy : {acc_ensemble_no_aug:.4f}")
    print(f"Precision: {prec_ensemble_no_aug:.4f}")
    print(f"Recall   : {rec_ensemble_no_aug:.4f}")
    print(f"F1-score : {f1_ensemble_no_aug:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_no_aug, y_pred_no_aug, target_names=class_names_no_aug, zero_division=0))

    cm_ensemble_no_aug = confusion_matrix(y_true_no_aug, y_pred_no_aug, labels=range(len(class_names_no_aug)))
    print("\nConfusion Matrix:")
    print(cm_ensemble_no_aug)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ensemble_no_aug, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_no_aug, yticklabels=class_names_no_aug)
    plt.title("MobileNetV2 (No Aug) 5-Fold Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
else:
    print("Could not perform ensemble evaluation for MobileNetV2 (No Aug) as no model probabilities were collected.")

# %% Cell 112
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# The `plot_training_curves` function is defined in a previous cell.
# We will ensure it's available here for clarity if this cell is run independently.
def plot_training_curves(histories, model_name):
    num_folds = len(histories)

    if num_folds == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(num_folds, 2, figsize=(15, 5 * num_folds))

    fig.suptitle(f'Training and Validation Curves for {model_name}', fontsize=16)

    for i, history in enumerate(histories):
        ax_loss = axes[i][0] if num_folds > 1 else axes[0]
        ax_loss.plot(history['train_loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Validation Loss')
        ax_loss.set_title(f'Fold {i+1} - Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        ax_acc = axes[i][1] if num_folds > 1 else axes[1]
        ax_acc.plot(history['train_acc'], label='Train Accuracy')
        ax_acc.plot(history['val_acc'], label='Validation Accuracy')
        ax_acc.set_title(f'Fold {i+1} - Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Define the path to the saved history file for MobileNetV2 (No Aug)
# gdrive_base_output_dir is assumed to be defined from previous cells
model_name_no_aug = 'mobilenetv2_lung_clean_no_aug'
load_history_path_no_aug = os.path.join(gdrive_base_output_dir, model_name_no_aug + "_histories_clean.json")

# Load the training history
try:
    with open(load_history_path_no_aug, 'r') as f:
        loaded_histories_no_aug = json.load(f)
    print(f"Successfully loaded training histories from: {load_history_path_no_aug}")

    # Display some information from the loaded histories
    print(f"\nNumber of folds in history: {len(loaded_histories_no_aug)}")
    for i, history in enumerate(loaded_histories_no_aug):
        print(f"  Fold {i+1} - Epochs trained: {len(history['train_loss'])}")
        print(f"  Fold {i+1} - Last Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"  Fold {i+1} - Last Val Acc: {history['val_acc'][-1]:.4f}")
        if 'lr' in history and len(history['lr']) > 0:
            print(f"  Fold {i+1} - Last LR: {history['lr'][-1]:.6f}")

    # Plot the training curves after loading histories
    plot_training_curves(loaded_histories_no_aug, model_name_no_aug)

except FileNotFoundError:
    print(f"Error: History file not found at {load_history_path_no_aug}. Please ensure training was completed and the file was saved.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {load_history_path_no_aug}. The file might be corrupted or empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# %% Cell 113
import json
import os
import numpy as np

# Define the output file path for the MobileNetV2 (No Aug) ensemble test results
output_file_path_no_aug = os.path.join(
    gdrive_base_output_dir,
    'mobilenetv2_lung_clean_no_aug_ensemble_test_results.json'
)

# Prepare the dictionary for JSON serialization
serializable_test_results_no_aug = {
    'accuracy': acc_ensemble_no_aug,
    'precision': prec_ensemble_no_aug,
    'recall': rec_ensemble_no_aug,
    'f1_score': f1_ensemble_no_aug,
    'confusion_matrix': cm_ensemble_no_aug.tolist(),
    'class_names': class_names_no_aug
}

# Save the test results to a JSON file
try:
    with open(output_file_path_no_aug, 'w') as f:
        json.dump(serializable_test_results_no_aug, f, indent=4)
    print(f"MobileNetV2 (No Aug) ensemble test results saved to: {output_file_path_no_aug}")
except Exception as e:
    print(f"An error occurred while saving test results: {e}")

# %% [markdown] Cell 114
# DenseNet121

# %% Cell 115
import os

# Temporarily override build_train_transform to remove all augmentations
# by using the validation/test transform (which only resizes and normalizes).
original_build_train_transform = build_train_transform
build_train_transform = build_val_test_transform

MODEL_SPECS_DENSENET121_NO_AUG = [
    {
        'model_to_load': "densenet121",
        'model_name': "densenet121_lung_clean_no_aug",
    }
]

os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

all_model_histories_densenet121_no_aug = {}

try:
    for model_spec in MODEL_SPECS_DENSENET121_NO_AUG:
        model_name = model_spec['model_name']
        print(f"\n{'='*50}")
        print(f"Initiating K-fold training for {model_name} WITHOUT augmentation")
        print(f"{'-'*50}")

        # Call the train_single_model_same_config function using identical folds_info
        histories = train_single_model_same_config(
            model_spec=model_spec,
            folds_info=folds_info,
            base_output_dir=gdrive_base_output_dir
        )

        all_model_histories_densenet121_no_aug[model_name] = histories

    print("\n" + "="*50)
    print("DenseNet121 (No Augmentation) model training completed.")
    print("="*50)
finally:
    # Restore the original transform function to avoid affecting subsequent runs
    build_train_transform = original_build_train_transform

# %% Cell 116
import numpy as np

model_name_densenet121_no_aug = 'densenet121_lung_clean_no_aug'
if 'all_model_histories_densenet121_no_aug' in locals() and model_name_densenet121_no_aug in all_model_histories_densenet121_no_aug:
    histories_d121_no_aug = all_model_histories_densenet121_no_aug[model_name_densenet121_no_aug]

    fold_best_accuracies = []
    for i, history in enumerate(histories_d121_no_aug):
        # Find the maximum validation accuracy achieved in each fold
        best_val_acc = np.max(history['val_acc'])
        fold_best_accuracies.append(best_val_acc)
        print(f"Fold {i+1} Best Validation Accuracy: {best_val_acc:.4f}")

    if fold_best_accuracies:
        avg_accuracy = np.mean(fold_best_accuracies)
        print(f"\nAverage Validation Accuracy across all folds (No Augmentation): {avg_accuracy:.4f}")
    else:
        print("No validation accuracies found.")
else:
    print("Error: Training histories not found in memory. Please ensure the model training was completed.")

# %% Cell 117
# Plot the training curves for DenseNet121 without augmentation
if 'histories_d121_no_aug' in locals():
    plot_training_curves(histories_d121_no_aug, model_name_densenet121_no_aug)

# %% Cell 118
model_architecture_densenet121_no_aug = 'densenet121'

# Get ensemble predictions
avg_probs_d121_no_aug, y_true_d121_no_aug, class_names_d121_no_aug = get_ensemble_predictions_for_model_type(
    model_architecture=model_architecture_densenet121_no_aug,
    model_name_prefix=model_name_densenet121_no_aug,
    num_folds=COMMON_CONFIG['num_folds'],
    base_output_dir_for_model=gdrive_base_output_dir,
    test_df=clean_test_df,
    common_config=COMMON_CONFIG,
    device=device,
    class_names=class_names
)

if avg_probs_d121_no_aug is not None:
    y_pred_d121_no_aug = np.argmax(avg_probs_d121_no_aug, axis=1)

    # Calculate metrics
    acc_ensemble_d121_no_aug = accuracy_score(y_true_d121_no_aug, y_pred_d121_no_aug)
    prec_ensemble_d121_no_aug = precision_score(y_true_d121_no_aug, y_pred_d121_no_aug, average="weighted", zero_division=0)
    rec_ensemble_d121_no_aug = recall_score(y_true_d121_no_aug, y_pred_d121_no_aug, average="weighted", zero_division=0)
    f1_ensemble_d121_no_aug = f1_score(y_true_d121_no_aug, y_pred_d121_no_aug, average="weighted", zero_division=0)

    print("\n========== FINAL CLEAN TEST RESULT: DenseNet121 (No Aug) 5-Fold Ensemble ==========")
    print(f"Accuracy : {acc_ensemble_d121_no_aug:.4f}")
    print(f"Precision: {prec_ensemble_d121_no_aug:.4f}")
    print(f"Recall   : {rec_ensemble_d121_no_aug:.4f}")
    print(f"F1-score : {f1_ensemble_d121_no_aug:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_d121_no_aug, y_pred_d121_no_aug, target_names=class_names_d121_no_aug, zero_division=0))

    cm_ensemble_d121_no_aug = confusion_matrix(y_true_d121_no_aug, y_pred_d121_no_aug, labels=range(len(class_names_d121_no_aug)))
    print("\nConfusion Matrix:")
    print(cm_ensemble_d121_no_aug)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ensemble_d121_no_aug, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_d121_no_aug, yticklabels=class_names_d121_no_aug)
    plt.title("DenseNet121 (No Aug) 5-Fold Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
else:
    print("Could not perform ensemble evaluation for DenseNet121 (No Aug) as no model probabilities were collected.")

# %% Cell 119
import json
import os
import numpy as np

filepath_d121_no_aug = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE/densenet121_lung_clean_no_aug_ensemble_test_results.json'

if os.path.exists(filepath_d121_no_aug):
    with open(filepath_d121_no_aug, 'r') as f:
        densenet121_no_aug_results = json.load(f)

    print("DenseNet121 (No Aug) 5-fold ensemble results:")
    for key, value in densenet121_no_aug_results.items():
        if key not in ['confusion_matrix', 'class_names']:
            print(f"{key}: {value}")

    print("\nClass Names:", densenet121_no_aug_results.get('class_names'))
    print("Confusion Matrix:")
    print(np.array(densenet121_no_aug_results.get('confusion_matrix')))
else:
    print(f"File not found: {filepath_d121_no_aug}")

# %% Cell 120
import json
import os

# Define the output file path for the DenseNet121 (No Aug) ensemble test results
output_file_path_d121_no_aug = os.path.join(
    gdrive_base_output_dir,
    'densenet121_lung_clean_no_aug_ensemble_test_results.json'
)

# Prepare the dictionary for JSON serialization and save
if 'acc_ensemble_d121_no_aug' in locals():
    serializable_test_results_d121_no_aug = {
        'accuracy': acc_ensemble_d121_no_aug,
        'precision': prec_ensemble_d121_no_aug,
        'recall': rec_ensemble_d121_no_aug,
        'f1_score': f1_ensemble_d121_no_aug,
        'confusion_matrix': cm_ensemble_d121_no_aug.tolist(),
        'class_names': class_names_d121_no_aug
    }

    try:
        with open(output_file_path_d121_no_aug, 'w') as f:
            json.dump(serializable_test_results_d121_no_aug, f, indent=4)
        print(f"DenseNet121 (No Aug) ensemble test results saved to: {output_file_path_d121_no_aug}")
    except Exception as e:
        print(f"An error occurred while saving test results: {e}")
else:
    print("Metrics for DenseNet121 (No Aug) not found. Please ensure the evaluation cell was run.")

# %% [markdown] Cell 121
# Dense201

# %% Cell 122
import os

# Temporarily override build_train_transform to remove all augmentations
# by using the validation/test transform (which only resizes and normalizes).
original_build_train_transform = build_train_transform
build_train_transform = build_val_test_transform

MODEL_SPECS_DENSENET201_NO_AUG = [
    {
        'model_to_load': "densenet201",
        'model_name': "densenet201_lung_clean_no_aug",
    }
]

os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

all_model_histories_densenet201_no_aug = {}

try:
    for model_spec in MODEL_SPECS_DENSENET201_NO_AUG:
        model_name = model_spec['model_name']
        print(f"\n{'='*50}")
        print(f"Initiating K-fold training for {model_name} WITHOUT augmentation")
        print(f"{'-'*50}")

        # Call the train_single_model_same_config function using identical folds_info
        histories = train_single_model_same_config(
            model_spec=model_spec,
            folds_info=folds_info,
            base_output_dir=gdrive_base_output_dir
        )

        all_model_histories_densenet201_no_aug[model_name] = histories

    print("\n" + "="*50)
    print("DenseNet201 (No Augmentation) model training completed.")
    print("="*50)
finally:
    # Restore the original transform function to avoid affecting subsequent runs
    build_train_transform = original_build_train_transform

# %% Cell 123
model_name_densenet201_no_aug = 'densenet201_lung_clean_no_aug'
model_architecture_densenet201_no_aug = 'densenet201'

# Get ensemble predictions
avg_probs_d201_no_aug, y_true_d201_no_aug, class_names_d201_no_aug = get_ensemble_predictions_for_model_type(
    model_architecture=model_architecture_densenet201_no_aug,
    model_name_prefix=model_name_densenet201_no_aug,
    num_folds=COMMON_CONFIG['num_folds'],
    base_output_dir_for_model=gdrive_base_output_dir,
    test_df=clean_test_df,
    common_config=COMMON_CONFIG,
    device=device,
    class_names=class_names
)

if avg_probs_d201_no_aug is not None:
    y_pred_d201_no_aug = np.argmax(avg_probs_d201_no_aug, axis=1)

    # Calculate metrics
    acc_ensemble_d201_no_aug = accuracy_score(y_true_d201_no_aug, y_pred_d201_no_aug)
    prec_ensemble_d201_no_aug = precision_score(y_true_d201_no_aug, y_pred_d201_no_aug, average="weighted", zero_division=0)
    rec_ensemble_d201_no_aug = recall_score(y_true_d201_no_aug, y_pred_d201_no_aug, average="weighted", zero_division=0)
    f1_ensemble_d201_no_aug = f1_score(y_true_d201_no_aug, y_pred_d201_no_aug, average="weighted", zero_division=0)

    print("\n========== FINAL CLEAN TEST RESULT: DenseNet201 (No Aug) 5-Fold Ensemble ==========")
    print(f"Accuracy : {acc_ensemble_d201_no_aug:.4f}")
    print(f"Precision: {prec_ensemble_d201_no_aug:.4f}")
    print(f"Recall   : {rec_ensemble_d201_no_aug:.4f}")
    print(f"F1-score : {f1_ensemble_d201_no_aug:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_d201_no_aug, y_pred_d201_no_aug, target_names=class_names_d201_no_aug, zero_division=0))

    cm_ensemble_d201_no_aug = confusion_matrix(y_true_d201_no_aug, y_pred_d201_no_aug, labels=range(len(class_names_d201_no_aug)))
    print("\nConfusion Matrix:")
    print(cm_ensemble_d201_no_aug)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ensemble_d201_no_aug, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_d201_no_aug, yticklabels=class_names_d201_no_aug)
    plt.title("DenseNet201 (No Aug) 5-Fold Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Save the test results to a JSON file
    output_file_path_d201_no_aug = os.path.join(
        gdrive_base_output_dir,
        'densenet201_lung_clean_no_aug_ensemble_test_results.json'
    )

    serializable_test_results_d201_no_aug = {
        'accuracy': acc_ensemble_d201_no_aug,
        'precision': prec_ensemble_d201_no_aug,
        'recall': rec_ensemble_d201_no_aug,
        'f1_score': f1_ensemble_d201_no_aug,
        'confusion_matrix': cm_ensemble_d201_no_aug.tolist(),
        'class_names': class_names_d201_no_aug
    }

    try:
        with open(output_file_path_d201_no_aug, 'w') as f:
            json.dump(serializable_test_results_d201_no_aug, f, indent=4)
        print(f"DenseNet201 (No Aug) ensemble test results saved to: {output_file_path_d201_no_aug}")
    except Exception as e:
        print(f"An error occurred while saving test results: {e}")

else:
    print("Could not perform ensemble evaluation for DenseNet201 (No Aug) as no model probabilities were collected.")

# %% Cell 124
import numpy as np
import torch
import json
import os
import timm

model_name_d201_no_aug = 'densenet201_lung_clean_no_aug'
model_architecture_d201_no_aug = 'densenet201'

# 1. Evaluate cumulative accuracy (average validation accuracy)
print("--- Cumulative Accuracy Evaluation ---")
histories_d201_no_aug = None
if 'all_model_histories_densenet201_no_aug' in locals() and model_name_d201_no_aug in all_model_histories_densenet201_no_aug:
    histories_d201_no_aug = all_model_histories_densenet201_no_aug[model_name_d201_no_aug]
else:
    history_path = os.path.join(gdrive_base_output_dir, f"{model_name_d201_no_aug}_histories_clean.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            histories_d201_no_aug = json.load(f)

if histories_d201_no_aug:
    fold_best_accuracies = [np.max(h['val_acc']) for h in histories_d201_no_aug]
    for i, acc in enumerate(fold_best_accuracies):
        print(f"Fold {i+1} Best Validation Accuracy: {acc:.4f}")
    print(f"\nAverage Validation Accuracy (Cumulative): {np.mean(fold_best_accuracies):.4f}")
else:
    print("Could not find histories for DenseNet201 (No Aug).")

# 2. Load the model and save as final model
print("\n--- Loading and Saving Final Model ---")
fold_to_load = 5
model_save_dir = os.path.join(gdrive_base_output_dir, model_name_d201_no_aug)
fold_weight_path = os.path.join(model_save_dir, f"{model_name_d201_no_aug}_fold_{fold_to_load}.pt")
fold_config_path = os.path.join(model_save_dir, f"{model_name_d201_no_aug}_fold_{fold_to_load}_config.json")

try:
    with open(fold_config_path, 'r') as f:
        fold_config = json.load(f)

    final_model_d201_no_aug = timm.create_model(
        model_architecture_d201_no_aug,
        pretrained=False,
        num_classes=fold_config.get('num_classes', 4)
    )
    final_model_d201_no_aug.load_state_dict(torch.load(fold_weight_path, map_location=device))
    final_model_d201_no_aug = final_model_d201_no_aug.to(device)
    print(f"Successfully loaded DenseNet201 (No Aug) model weights from: {fold_weight_path}")

    final_save_path = os.path.join(gdrive_base_output_dir, f"{model_name_d201_no_aug}_final_model.pt")
    torch.save(final_model_d201_no_aug.state_dict(), final_save_path)
    print(f"Saved final DenseNet201 (No Aug) model to: {final_save_path}")
except Exception as e:
    print(f"Error loading or saving model: {e}")

# %% Cell 125
import torch
import timm
import os

# Define model details
model_name_d201_no_aug = 'densenet201_lung_clean_no_aug'
model_architecture_d201_no_aug = 'densenet201'

# Path to the saved final model
final_model_path = os.path.join(gdrive_base_output_dir, f"{model_name_d201_no_aug}_final_model.pt")

if os.path.exists(final_model_path):
    print(f"Loading final model from: {final_model_path}")

    # Re-create the model architecture
    loaded_final_model = timm.create_model(
        model_architecture_d201_no_aug,
        pretrained=False,
        num_classes=num_classes
    )

    # Load the saved state dictionary
    loaded_final_model.load_state_dict(torch.load(final_model_path, map_location=device))
    loaded_final_model = loaded_final_model.to(device)
    loaded_final_model.eval() # Set to evaluation mode

    print("Model successfully loaded and set to evaluation mode.")
else:
    print(f"Error: Final model not found at {final_model_path}")

# %% Cell 126
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Temporarily override build_train_transform to remove all augmentations
# by using the validation/test transform (which only resizes and normalizes).
original_build_train_transform = build_train_transform
build_train_transform = build_val_test_transform

# Define the model specifications for EfficientNetV2-S without augmentation
MODEL_SPECS_EFFICIENTNETV2_S_NO_AUG = [
    {
        'model_to_load': 'tf_efficientnetv2_s',
        'model_name': 'efficientnetv2_s_lung_clean_no_aug',
    }
]

os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

# Dictionary to store training histories for all models
all_model_histories_efficientnetv2_s_no_aug = {}

# NOTE: We are using a reduced batch size of 8 (down from 16).
# This is why you see 68 steps (542 images / 8) instead of 34 steps (542 images / 16).
modified_folds_info = []
reduced_batch_size = 16

print(f"\nTemporarily reducing batch size to {reduced_batch_size} for EfficientNetV2-S (No Augmentation).")
print(f"This increases the number of steps per epoch to ensure GPU memory stability.")

for f_info in folds_info:
    temp_config = f_info['config'].copy()
    temp_config['batch_size'] = reduced_batch_size
    modified_folds_info.append({
        'fold': f_info['fold'],
        'train_df': f_info['train_df'],
        'val_df': f_info['val_df'],
        'config': temp_config
    })

try:
    for model_spec in MODEL_SPECS_EFFICIENTNETV2_S_NO_AUG:
        model_name = model_spec['model_name']
        print(f"\n{'='*50}")
        print(f"Initiating K-fold training for {model_name} WITHOUT augmentation")
        print(f"{'-'*50}")

        histories = train_single_model_same_config(
            model_spec=model_spec,
            folds_info=modified_folds_info,
            base_output_dir=gdrive_base_output_dir
        )

        all_model_histories_efficientnetv2_s_no_aug[model_name] = histories

    print("\n" + "="*50)
    print("EfficientNetV2-S (No Augmentation) model training completed.")
    print("="*50)

    model_name_efficientnetv2_s_no_aug = 'efficientnetv2_s_lung_clean_no_aug'
    if model_name_efficientnetv2_s_no_aug in all_model_histories_efficientnetv2_s_no_aug:
        histories_e2v2_s_no_aug = all_model_histories_efficientnetv2_s_no_aug[model_name_efficientnetv2_s_no_aug]
        print(f"\nPlotting training curves for {model_name_efficientnetv2_s_no_aug}:")
        plot_training_curves(histories_e2v2_s_no_aug, model_name_efficientnetv2_s_no_aug)

    print("\n--- Cumulative Average Validation Accuracy ---")
    if model_name_efficientnetv2_s_no_aug in all_model_histories_efficientnetv2_s_no_aug:
        histories_e2v2_s_no_aug = all_model_histories_efficientnetv2_s_no_aug[model_name_efficientnetv2_s_no_aug]
        fold_best_accuracies = [np.max(h['val_acc']) for h in histories_e2v2_s_no_aug]
        for i, acc in enumerate(fold_best_accuracies):
            print(f"Fold {i+1} Best Validation Accuracy: {acc:.4f}")
        print(f"\nAverage Validation Accuracy (Cumulative): {np.mean(fold_best_accuracies):.4f}")

finally:
    build_train_transform = original_build_train_transform
    print("\nOriginal build_train_transform function restored.")

# %% [markdown] Cell 127
# Moilenet_V3 large

# %% Cell 128
import os

# Temporarily override build_train_transform to remove all augmentations
# by using the validation/test transform (which only resizes and normalizes).
original_build_train_transform = build_train_transform
build_train_transform = build_val_test_transform

MODEL_SPECS_MOBILENETV3_LARGE_NO_AUG = [
    {
        'model_to_load': "mobilenetv3_large_100",
        'model_name': "mobilenetv3_large_100_lung_clean_no_aug",
    }
]

os.makedirs(gdrive_base_output_dir, exist_ok=True)
print(f"Using base output directory for model outputs: {gdrive_base_output_dir}")

all_model_histories_mobilenetv3_large_no_aug = {}

try:
    for model_spec in MODEL_SPECS_MOBILENETV3_LARGE_NO_AUG:
        model_name = model_spec['model_name']
        print(f"\n{'='*50}")
        print(f"Initiating K-fold training for {model_name} WITHOUT augmentation")
        print(f"{'-'*50}")

        # Call the train_single_model_same_config function using identical folds_info
        histories = train_single_model_same_config(
            model_spec=model_spec,
            folds_info=folds_info,
            base_output_dir=gdrive_base_output_dir
        )

        all_model_histories_mobilenetv3_large_no_aug[model_name] = histories

    print("\n" + "="*50)
    print("MobileNetV3 Large 100 (No Augmentation) model training completed.")
    print("="*50)
finally:
    # Restore the original transform function to avoid affecting subsequent runs
    build_train_transform = original_build_train_transform

# %% [markdown] Cell 129
# Commulative accuracy

# %% Cell 130
import numpy as np

model_name_mobilenetv3_large_no_aug = 'mobilenetv3_large_100_lung_clean_no_aug'
if 'all_model_histories_mobilenetv3_large_no_aug' in locals() and model_name_mobilenetv3_large_no_aug in all_model_histories_mobilenetv3_large_no_aug:
    histories_mobilenetv3_large_no_aug = all_model_histories_mobilenetv3_large_no_aug[model_name_mobilenetv3_large_no_aug]

    fold_best_accuracies = []
    for i, history in enumerate(histories_mobilenetv3_large_no_aug):
        # Find the maximum validation accuracy achieved in each fold
        best_val_acc = np.max(history['val_acc'])
        fold_best_accuracies.append(best_val_acc)
        print(f"Fold {i+1} Best Validation Accuracy: {best_val_acc:.4f}")

    if fold_best_accuracies:
        avg_accuracy = np.mean(fold_best_accuracies)
        print(f"\nAverage Validation Accuracy across all folds (No Augmentation): {avg_accuracy:.4f}")
    else:
        print("No validation accuracies found.")
else:
    print("Error: Training histories not found in memory. Please ensure the model training was completed.")

# %% [markdown] Cell 131
# Ensamble Accuracy

# %% Cell 132
import os
import json
import numpy as np
import torch
import warnings
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_ensemble_predictions_for_model_type(
    model_architecture,
    model_name_prefix,
    num_folds,
    base_output_dir_for_model,
    test_df,
    common_config,
    device,
    class_names
):
    print(f"\n--- Loading and collecting predictions for {model_name_prefix} (5-fold ensemble) ---")
    loaded_fold_models = {}

    for fold_number in range(1, num_folds + 1):
        model_save_dir = os.path.join(base_output_dir_for_model, model_name_prefix)
        fold_weight_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}.pt"
        )
        fold_config_path = os.path.join(
            model_save_dir,
            f"{model_name_prefix}_fold_{fold_number}_config.json"
        )

        try:
            with open(fold_config_path, "r") as f:
                fold_config = json.load(f)

            fold_config['mean_overall'] = np.array(fold_config['mean_overall']) if isinstance(fold_config['mean_overall'], list) else fold_config['mean_overall']
            fold_config['std_overall'] = np.array(fold_config['std_overall']) if isinstance(fold_config['std_overall'], list) else fold_config['std_overall']

            model = build_timm_model(
                model_architecture,
                fold_config
            )
            model.load_state_dict(
                torch.load(fold_weight_path, map_location=device)
            )
            model = model.to(device)
            model.eval()

            loaded_fold_models[f'fold_{fold_number}'] = {
                'model': model,
                'config': fold_config
            }
        except FileNotFoundError:
            print(f"Error: Weights or config not found for {model_name_prefix} Fold {fold_number} at {fold_weight_path} or {fold_config_path}")
            continue
        except Exception as e:
            print(f"An error occurred while loading {model_name_prefix} Fold {fold_number}: {e}")
            continue

    if not loaded_fold_models:
        print(f"No models loaded for {model_name_prefix}. Cannot perform ensemble predictions.")
        return None, None, None

    all_fold_probs = []
    y_true_ret = None

    print(f"Collecting test predictions for {model_name_prefix}...")
    for fold_number in range(1, num_folds + 1):
        model_data = loaded_fold_models.get(f'fold_{fold_number}')
        if not model_data:
            continue

        model = model_data['model']
        fold_config = model_data['config']

        transform_test = build_val_test_transform(fold_config)

        test_dataset = CustomImageDataset(
            test_df,
            transform=transform_test,
            labels_map=fold_config['labels_map']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=common_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        fold_probs = []
        fold_labels = []

        with torch.inference_mode():
            for inputs, labels in tqdm(test_loader, desc=f"{model_name_prefix} Fold {fold_number} Test Predictions"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                fold_probs.extend(probs.cpu().numpy())
                fold_labels.extend(labels.numpy())

        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)
        all_fold_probs.append(fold_probs)

        if y_true_ret is None:
            y_true_ret = fold_labels
        else:
            if not np.array_equal(y_true_ret, fold_labels):
                warnings.warn(f"Labels for {model_name_prefix} changed between folds, this should not happen with fixed random seed.")

    if not all_fold_probs:
        print(f"No predictions collected for {model_name_prefix}.")
        return None, None, None

    avg_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    return avg_probs, y_true_ret, class_names

model_name_mobilenetv3_large_no_aug = 'mobilenetv3_large_100_lung_clean_no_aug'
model_architecture_mobilenetv3_large_no_aug = 'mobilenetv3_large_100'

# Get ensemble predictions
avg_probs_mv3_no_aug, y_true_mv3_no_aug, class_names_mv3_no_aug = get_ensemble_predictions_for_model_type(
    model_architecture=model_architecture_mobilenetv3_large_no_aug,
    model_name_prefix=model_name_mobilenetv3_large_no_aug,
    num_folds=COMMON_CONFIG['num_folds'],
    base_output_dir_for_model=gdrive_base_output_dir,
    test_df=clean_test_df,
    common_config=COMMON_CONFIG,
    device=device,
    class_names=class_names
)

if avg_probs_mv3_no_aug is not None:
    y_pred_mv3_no_aug = np.argmax(avg_probs_mv3_no_aug, axis=1)

    # Calculate metrics
    acc_ensemble_mv3_no_aug = accuracy_score(y_true_mv3_no_aug, y_pred_mv3_no_aug)
    prec_ensemble_mv3_no_aug = precision_score(y_true_mv3_no_aug, y_pred_mv3_no_aug, average="weighted", zero_division=0)
    rec_ensemble_mv3_no_aug = recall_score(y_true_mv3_no_aug, y_pred_mv3_no_aug, average="weighted", zero_division=0)
    f1_ensemble_mv3_no_aug = f1_score(y_true_mv3_no_aug, y_pred_mv3_no_aug, average="weighted", zero_division=0)

    print("\n========== FINAL CLEAN TEST RESULT: MobileNetV3 Large (No Aug) 5-Fold Ensemble ==========")
    print(f"Accuracy : {acc_ensemble_mv3_no_aug:.4f}")
    print(f"Precision: {prec_ensemble_mv3_no_aug:.4f}")
    print(f"Recall   : {rec_ensemble_mv3_no_aug:.4f}")
    print(f"F1-score : {f1_ensemble_mv3_no_aug:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_mv3_no_aug, y_pred_mv3_no_aug, target_names=class_names_mv3_no_aug, zero_division=0))

    cm_ensemble_mv3_no_aug = confusion_matrix(y_true_mv3_no_aug, y_pred_mv3_no_aug, labels=range(len(class_names_mv3_no_aug)))
    print("\nConfusion Matrix:")
    print(cm_ensemble_mv3_no_aug)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ensemble_mv3_no_aug, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_mv3_no_aug, yticklabels=class_names_mv3_no_aug)
    plt.title("MobileNetV3 Large (No Aug) 5-Fold Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Save the test results to a JSON file
    output_file_path_mv3_no_aug = os.path.join(
        gdrive_base_output_dir,
        'mobilenetv3_large_100_lung_clean_no_aug_ensemble_test_results.json'
    )

    serializable_test_results_mv3_no_aug = {
        'accuracy': acc_ensemble_mv3_no_aug,
        'precision': prec_ensemble_mv3_no_aug,
        'recall': rec_ensemble_mv3_no_aug,
        'f1_score': f1_ensemble_mv3_no_aug,
        'confusion_matrix': cm_ensemble_mv3_no_aug.tolist(),
        'class_names': class_names_mv3_no_aug
    }

    try:
        with open(output_file_path_mv3_no_aug, 'w') as f:
            json.dump(serializable_test_results_mv3_no_aug, f, indent=4)
        print(f"MobileNetV3 Large (No Aug) ensemble test results saved to: {output_file_path_mv3_no_aug}")
    except Exception as e:
        print(f"An error occurred while saving test results: {e}")

else:
    print("Could not perform ensemble evaluation for MobileNetV3 Large (No Aug) as no model probabilities were collected.")

# %% Cell 134
import os
import json
import glob
import pandas as pd

gdrive_base_output_dir = '/content/gdrive/MyDrive/DL_Model_Outputs_CLEAN_NO_LEAKAGE'

# Glob for all test result files
result_files = glob.glob(os.path.join(gdrive_base_output_dir, '*test_results.json'))

all_results = []

for filepath in result_files:
    filename = os.path.basename(filepath)

    # Format model name
    model_name = filename.replace('_ensemble_test_results.json', ' (Ensemble)')
    model_name = model_name.replace('_cumulative_test_results.json', ' (Cumulative)')

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Some files might use 'average_accuracy'
        acc = data.get('accuracy', data.get('average_accuracy', None))
        prec = data.get('precision', None)
        rec = data.get('recall', None)
        f1 = data.get('f1_score', None)

        if acc is not None:
            all_results.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1 Score': f1
            })
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if all_results:
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    print("========== ALL MODELS ACCURACY EVALUATION ==========")
    display(df_results)
else:
    print("No test result files found in the output directory.")
