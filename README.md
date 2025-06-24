# Project-2-Music-Genre-Identification

# Music Genre Classification with Optimized CNN

## Project Overview

This repository contains an end-to-end pipeline for music genre classification using an optimized Convolutional Neural Network (CNN). The project utilizes the GTZAN dataset (10 genres, 1000 audio tracks) and leverages advanced audio feature extraction and deep learning techniques to classify genres from audio files.

---

## Data Summary

- **Dataset:** GTZAN (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **Samples:** 1000 audio tracks (`.au` format), 100 per genre
- **Features:** Mel-spectrogram, MFCC, chroma, spectral contrast (stacked and padded)
- **Input Shape:** (187, 130, 1)
- **Split:** 80% training, 20% validation

---

## Model Architecture

- **Type:** 2D Convolutional Neural Network (CNN)
- **Layers:** 
  - 3 × Conv2D + MaxPooling2D + BatchNorm + Dropout
  - GlobalAveragePooling2D
  - Dense(256) + BatchNorm + Dropout
  - Output: Dense(10, softmax)
- **Regularization:** BatchNorm, Dropout, L2 regularization
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

---

## Training & Evaluation

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

## Hardware Constraints

> **Note:**  
> The performance of this model was also limited by the available GPU VRAM. Due to VRAM constraints, batch size and model complexity had to be reduced, which impacted both training speed and achievable accuracy. Larger batch sizes or deeper models could not be used without causing out-of-memory errors, a common bottleneck in deep learning workflows on consumer GPUs[2][3][5].

---

## How to Run

1. **Install Requirements**

2. **Prepare Data**
- Place the GTZAN dataset in the `genres/` directory as structured in the notebook.

3. **Run the Notebook**
- Open `Project2_CNN_optimised.ipynb` in Jupyter or VSCode.
- Execute all cells in order.

4. **Model Output**
- Trained model saved as `optimized_music_genre_cnn.h5`
- Label encoder saved as `label_encoder.pkl`

---

## Notable Techniques

- **Feature Fusion:** Combines mel-spectrogram, MFCC, chroma, and spectral contrast for richer input.
- **Data Normalization & Padding:** Ensures consistent input shapes.
- **Class Weighting:** Handles class imbalance.
- **Regularization:** Dropout and BatchNorm reduce overfitting.
- **Early Stopping & LR Scheduling:** Prevents overfitting and accelerates convergence.

---

## License

This repository is for academic use only.

---

## Acknowledgements

- GTZAN dataset
- Librosa, Keras, TensorFlow, scikit-learn

---

**For details, see the notebook: `Project2_CNN_optimised.ipynb`**  
