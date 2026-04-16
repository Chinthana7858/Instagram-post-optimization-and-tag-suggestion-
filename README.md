# Research Project

## Project Overview

This repository contains a social media hashtag prediction and reach optimization research project. It includes data preprocessing, image-only modeling, multimodal modeling with user influence, and experimental optimizations for hashtag reach. The code is mainly implemented in Python and uses PyTorch, torchvision, scikit-learn, pandas, and related libraries.

> Note: Most scripts currently use hardcoded Windows paths such as `E:\FinalData` and `E:\Dataset`. Update these paths before running the code on another machine.
<img width="1408" height="969" alt="image" src="https://github.com/user-attachments/assets/5ae180df-2f92-4f7f-820a-a4046081a70c" />

---

## Directory Structure

- `image_only/`
- `multimodel_Resnet50/`
  - `mulrimodel_Resnet50/`
  - `UserInfluence&Co_Occurence/`
  - `WithReachOptimize/`
- `multimodel_vits/`
- `preprocess/`
- `single_preprocess/`
- `username_encoder.pkl`

---

## Folder and File Usage

### `image_only/`

Contains an image-only ResNet-50 classification training pipeline.

- `training.py`
  - Trains a ResNet-50 model using image data only.
  - Uses a multilabel classification setup for hashtag prediction.
  - Loads image files and CSV labels, trains the model, evaluates on validation data, and plots losses and F1 scores.

### `multimodel_Resnet50/`

Contains multimodal ResNet-50 experiments combining image input with user metadata.

#### `multimodel_Resnet50/mulrimodel_Resnet50/`

- `multimodel_Resnet50.py`
  - Main multimodal model using ResNet-50 plus user embeddings.
  - Builds a dataset with image and per-user feature vectors.
  - Trains and validates a joint model for hashtag prediction.
- `pth_to_onnx.py`
  - Likely exports trained PyTorch `.pth` models to ONNX format.

#### `multimodel_Resnet50/UserInfluence&Co_Occurence/`

- `multimodel_Resnet50_with_user_influence_and_co_occurance.py`
  - Adds user influence and hashtag co-occurrence modeling to the multimodal ResNet pipeline.
  - Uses a co-occurrence matrix head and user-weighted embedding to improve predictions.

#### `multimodel_Resnet50/WithReachOptimize/`

This folder contains reach optimization experiments designed to jointly optimize hashtag prediction and a reach score metric.

##### `LossFunction/`

- `multimodel_Resnet50_with_Reach_optimize.py`
  - Trains a ResNet-based multimodal model that includes an IHC reach score component (`IHC_h`).
  - Implements a combined loss using tag prediction plus reach prediction.
- `pth_to_onnx.py`
  - Exports reach-optimized models to ONNX.

##### `SeperateModels/`

Contains separate model experiments, evaluation artifacts, and versioned optimizations.

- `ihc_model_train.py`
  - Likely trains a specialized reach-prediction model using IHC data.
- `multimodel_with_reach_optimization.py`
  - Another version of the reach optimization training pipeline.
- `best_model.pth`
  - Saved model weights from training.
- `ihc_scaler.pkl`
  - Saved scaler for reach-related feature normalization.

###### `SeperateModels/FinalModel/`

- `final_evaluation.py`
  - Evaluation script for the final reach-optimized pipeline.
- `ihc_optimization_summary.csv`
  - Evaluation summary data for reach optimization.
- `prediction.py`
  - Likely runs inference to predict hashtags or reach.
- `Reach_predicter.py`
  - Reach prediction utility.

###### `SeperateModels/v2/`

- `ihc_model_train.py`
- `multimodel_with_reach_optimization.py`

Version 2 of the separate reach optimization experiments.

###### `SeperateModels/v3/`

Contains a more advanced version with evaluation artifacts and export utilities.

- `final_evaluation.py`
- `ihc_model_train.py`
- `ihc_optimization_evaluation.csv`
- `ihc_optimization_summary.csv`
- `multimodel_with_reach_optimization.py`
- `to_onnx.py`
- `updated_reach_predicter.py`
- Multiple visualization PNGs and saved scaler/model files.

### `multimodel_vits/`

Contains a multimodal Vision Transformer (ViT) model experiment.

- `multimodel_vits.py`
  - Uses `timm` to load a ViT backbone and combines image features with user embeddings.
  - Trains and evaluates a multimodal hashtag prediction model using ViT instead of ResNet.

### `preprocess/`

Contains a sequential preprocessing pipeline for building the dataset from raw influencer and post metadata.

- `1_filter_users.py`
  - Samples or filters influencer users by category.
- `2_select_posts.py`
  - Selects posts for the chosen users and saves mapping files.
- `3_extract_post_data.py`
  - Extracts relevant post metadata from JSON files.
- `4_clean_posts.py`
  - Cleans captions and hashtags.
- `5_make_images_rows.py`
  - Converts posts into image rows suitable for modeling.
- `6_calculate_reach_contribution.py`
  - Computes a reach score per post.
- `7_move_images_to_folder.py`
  - Copies selected image files into a target folder.
- `8_get_all_hashtags.py`
  - Extracts hashtags from the dataset.
- `9_User_Hashtag_Frequency_Matrix.py`
  - Builds a per-user hashtag frequency matrix used for multimodal inputs.

### `single_preprocess/`

Contains exploratory and cleaning tools for a single preprocessing pipeline.

- `analysis.py`
  - Generates data analysis and reporting visualizations, saving a PDF report.
- `clean_data.py`
  - Removes duplicates and caps numeric outliers using IQR.
- `evaluate_dataset.py`
  - Likely evaluates dataset statistics (not detailed here, but part of dataset review).
- `preprocess.py`
  - A full end-to-end preprocessing workflow from influencer selection to reach computation.
  - Steps include selecting users, selecting posts, enriching posts with JSON metadata, cleaning hashtags, expanding image-tag rows, computing reach, copying selected images, generating top hashtags, and building a user-hashtag matrix.

### Root files

- `username_encoder.pkl`
  - Serialized artifact used for encoding usernames or user features.
  - No script in the repository references it directly in the files we inspected, but it is likely used in model input preprocessing.

---

## How to Use

1. Update dataset paths in scripts
   - Most scripts hardcode paths like `E:\FinalData`, `E:\Dataset`, or `E:\ProcessedV3`.
   - Change these paths to your local dataset locations before running.

2. Prepare data
   - Use `single_preprocess/preprocess.py` or the `preprocess/` step scripts to build CSV datasets and copy images.
   - Ensure required CSV files and image folders exist.

3. Train models
   - Run the desired model script from its folder. Example:
     - `python image_only/training.py`
     - `python multimodel_Resnet50/mulrimodel_Resnet50/multimodel_Resnet50.py --mode train`
     - `python multimodel_vits/multimodel_vits.py --mode train`
     - `python multimodel_Resnet50/UserInfluence&Co_Occurence/multimodel_Resnet50_with_user_influence_and_co_occurance.py --mode train`
     - `python multimodel_Resnet50/WithReachOptimize/LossFunction/multimodel_Resnet50_with_Reach_optimize.py --mode train`

4. Evaluate / test
   - Use `--mode test` where supported.
   - For scripts without mode flags, inspect the end of the file or add a small wrapper to run evaluation.

---

## Dependencies

Based on imports across the project, expected Python dependencies include:

- Python 3.8+ (recommended)
- torch
- torchvision
- timm
- pandas
- numpy
- scikit-learn
- pillow
- tqdm
- matplotlib
- seaborn

Install dependencies with pip, for example:

```bash
pip install torch torchvision timm pandas numpy scikit-learn pillow tqdm matplotlib seaborn
```

---

## Notes

- This repository is research-focused and contains multiple experimental versions.
- Several folders contain incremental versions (`v2`, `v3`) and saved artifacts.
- Most scripts are self-contained and can be adapted by changing config paths and model save paths.
- If you want to standardize the project, the next step is to centralize configuration into a single `config.py` and remove hardcoded paths.
