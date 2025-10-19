# Brain Tumor Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify brain tumor MRI images into four categories: glioma, meningioma, no tumor, and pituitary.

## Dataset

The dataset used is the **Brain Tumor MRI Dataset** from Kaggle, containing 5712 training images and 1311 testing images across four classes.

### Class Distribution:
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Model Architecture

The CNN model consists of:
- 6 Convolutional layers with MaxPooling
- 2 Fully Connected (Dense) layers
- Output layer with 4 units (softmax activation)

### Layer Details:
- Input: 256×256×3 RGB images
- Conv2D layers with increasing filters (64 → 128 → 256)
- MaxPooling2D after each Conv layer
- Flatten layer
- Dense layer with 256 units (ReLU activation)
- Output layer with 4 units (Softmax activation)

## Training

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 32 (default)
- **Validation Split**: 50% of test data

## Results

After 10 epochs of training:
- **Training Accuracy**: ~98.8%
- **Validation Accuracy**: ~92.3%
- **Test Metrics**:
  - Precision: 87.1%
  - Recall: 87.1%
  - Accuracy: 93.5%
