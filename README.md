# Project-2-Music-Genre-Identification

# Music Genre Classification with Deep Learning: Model Attempts & Results

## Project Overview

This repository contains an end-to-end pipeline for music genre classification using the GTZAN dataset (10 genres, 1000 audio tracks). Multiple deep learning approaches were explored, including classic 2D CNNs, 1D CNNs on raw audio, and hybrid CNN-Transformer models. The goal was to compare architectures, understand their strengths/limitations, and optimize for best accuracy within hardware constraints.

---

## Data Summary

- **Dataset:** GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **Samples:** 1000 audio tracks (`.au` format), 100 per genre
- **Features:** Mel-spectrogram, MFCC, chroma, spectral contrast (stacked and padded for some models)
- **Input Shapes:**  
  - 2D CNN/Transformer: `(187, 130, 1)` or similar  
  - 1D CNN: `(110250, 1)` (raw waveform, 5 seconds at 22050 Hz)
- **Split:** 80% training, 20% validation

---

## Model Attempts and Outcomes

### 1. **2D CNN on MFCC Features**
- **Approach:** Used MFCC features padded to `(40, 130)` and a simple 2D CNN with two Conv2D layers, MaxPooling, Flatten, and Dense layers.
- **Result:**  
  - **Validation Accuracy:** ~35%  
  - **Macro F1:** ~0.33  
- **Limitation:**  
  - Model overfit quickly (train acc >> val acc), and validation accuracy plateaued.
  - Only MFCCs used; richer features may be needed.
  - Model was limited in complexity to avoid VRAM issues.

### 2. **2D CNN with Feature Fusion (Mel, MFCC, Chroma, Contrast)**
- **Approach:** Combined multiple features into a stacked input, used a deeper 2D CNN with BatchNorm, Dropout, and L2 regularization.
- **Result:**  
  - **Validation Accuracy:** 61%  
  - **Macro F1:** 0.59  
  - **Best Genres:** Classical (F1 0.90), Metal (F1 0.73), Pop (F1 0.70)  
  - **Challenging Genres:** Disco, Rock
- **Limitation:**  
  - **GPU VRAM:** Could not use larger batch sizes or deeper models due to out-of-memory errors.
  - **Training Speed:** Training was slow with batch size reduced to 8.
  - **Further scaling (e.g., EfficientNet, more layers) was not possible on available hardware.**

### 3. **1D CNN on Raw Audio**
- **Approach:** Used a deep 1D CNN (M5-style) directly on normalized raw waveform segments.
- **Result:**  
  - **Validation Accuracy:** Peaked at ~17%  
  - **Observation:** Model struggled to learn meaningful representations from raw audio with limited data and compute.
- **Limitation:**  
  - **Data Representation:** Raw audio proved much harder to train on than spectrograms.
  - **VRAM:** Large input size (110,250 samples per clip) further limited batch size and model depth.

### 4. **Hybrid CNN-Transformer Model**
- **Approach:** Combined Conv2D feature extraction with Transformer encoder blocks on stacked features.
- **Result:**  
  - **Validation Accuracy:** Stuck at 10% (random guess)  
  - **Observation:** Model failed to learn; likely due to insufficient data, over-parameterization, or sub-optimal hyperparameters.
- **Limitation:**  
  - **System:** Transformer blocks require more VRAM and are sensitive to input shape and batch size.
  - **Data:** 1000 samples may be insufficient for Transformer-based models without augmentation or pretraining.

---

## Hardware Constraints

> **Note:**  
> The performance of all models was limited by the available GPU VRAM (NVIDIA consumer GPU, 8GB). Due to VRAM constraints, batch size and model complexity had to be reduced, which impacted both training speed and achievable accuracy. Larger batch sizes or deeper models could not be used without causing out-of-memory errors, a common bottleneck in deep learning workflows on consumer GPUs.

---

## Training & Evaluation (Best Model)

- **Epochs:** 50 (with early stopping)
- **Batch Size:** 8
- **Class Weights:** Used for balanced training

### **Results (Validation Set)**
| Genre      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| blues      | 0.59      | 0.65   | 0.62     | 20      |
| classical  | 0.86      | 0.95   | 0.90     | 20      |
| country    | 0.57      | 0.40   | 0.47     | 20      |
| disco      | 0.60      | 0.15   | 0.24     | 20      |
| hiphop     | 0.55      | 0.80   | 0.65     | 20      |
| jazz       | 0.60      | 0.45   | 0.51     | 20      |
| metal      | 0.62      | 0.90   | 0.73     | 20      |
| pop        | 0.62      | 0.80   | 0.70     | 20      |
| reggae     | 0.67      | 0.70   | 0.68     | 20      |
| rock       | 0.41      | 0.35   | 0.38     | 20      |

- **Overall Accuracy:** **61%**
- **Macro F1-score:** **0.59**
- **Best Genres:** Classical (F1 0.90), Metal (F1 0.73), Pop (F1 0.70)
- **Challenging Genres:** Disco, Rock

---

## Summary Table: Model Attempts

| Model/Features         | Max Val Acc | Macro F1 | Main Limitation                |
|------------------------|-------------|----------|-------------------------------|
| 2D CNN (MFCC)          | 35%         | 0.33     | Overfit, limited features     |
| 2D CNN (Feature Fusion)| 61%         | 0.59     | VRAM, batch size, speed       |
| 1D CNN (Raw Audio)     | 17%         | 0.15     | Data, VRAM, representation    |
| CNN-Transformer Hybrid | 10%         | 0.10     | Data, VRAM, model size        |

---

## How to Run

1. **Install Requirements**

pip install -r requirements.txt


2. **Prepare Data**
- Place the GTZAN dataset in the `genres/` directory as structured in the notebook.

3. **Run the Notebook**
- Open the desired notebook (e.g., `Project2_submission.ipynb`, `Project2_1D_CNN.ipynb`, or `Project2_Transformers.ipynb`) in Jupyter or VSCode.
- Execute all cells in order.

4. **Model Output**
- Trained models saved as `.h5` files
- Label encoder saved as `label_encoder.pkl`

---

## Recommendations & Future Work

- **Data Augmentation:** Add pitch/time shift, noise, or mixup for more robust models.
- **Transfer Learning:** Use pre-trained audio models (e.g., VGGish, YAMNet) for feature extraction.
- **Ensemble Methods:** Combine predictions from different models.
- **More VRAM:** Training on larger GPUs would allow deeper models and larger batch sizes for better performance.
- **Larger Datasets:** Transformers and 1D CNNs especially benefit from more data.

---

**For details, see the notebooks: `Project2_submission.ipynb`, `Project2_1D_CNN.ipynb`, `Project2_Transformers.ipynb`**
