# CAPTCHA Recognition Model

This repository contains a PyTorch-based implementation of a Convolutional Neural Network (CNN) for recognizing CAPTCHA images. The model processes grayscale CAPTCHA images, predicts the characters they contain, and can generalize well to unseen CAPTCHA samples.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Code Explanation](#code-explanation)
3. [Dataset Format](#dataset-format)
4. [Setup and Usage](#setup-and-usage)
5. [Results](#results)
6. [Future Improvements](#future-improvements)
7. [License](#license)

---

## Model Architecture

The CAPTCHA recognition model is designed to predict a sequence of characters from a CAPTCHA image. It is a combination of CNN for feature extraction and fully connected layers for sequence prediction.

### Model Components

1. **Convolutional Layers**:
   - Extract spatial features from the input image.
   - Three convolutional blocks with the following layers:
     - Convolution (`Conv2d`) with increasing filter sizes (32, 64, 128).
     - Activation (`ReLU`).
     - Downsampling (`MaxPool2d`).

2. **Fully Connected Layers**:
   - Flatten the output of the convolutional layers.
   - First layer maps extracted features to a 1024-dimensional vector.
   - Second layer outputs a tensor of shape `(batch_size, 5, 36)` to predict the 5-character CAPTCHA.

3. **Output Structure**:
   - The model predicts each character as a one-hot vector of length 36 (10 digits + 26 lowercase letters).

---

## Code Explanation

The code is structured into multiple stages:

### 1. **Dataset Preparation**
- `CaptchaDataset`: A custom PyTorch dataset class.
  - Reads CAPTCHA images from the `dataset` folder.
  - Converts images to grayscale, resizes them to `128x64`, and normalizes pixel values.
  - Extracts the text label from the image filename.
  - One-hot encodes the label for supervised learning.

### 2. **Model Definition**
- `CaptchaModel`: The main PyTorch model class.
  - Contains convolutional and fully connected layers.
  - Outputs a tensor reshaped for 5-character sequence prediction.

### 3. **Training Loop**
- Implements a training loop with the following steps:
  1. Forward pass: Input images through the model.
  2. Compute loss using `BCEWithLogitsLoss` for multi-label classification.
  3. Backpropagate gradients and update weights using the Adam optimizer.
- Logs training loss at each epoch.

### 4. **Evaluation**
- Decodes predictions by mapping one-hot vectors back to character labels.
- Compares predicted labels with ground truth to compute accuracy.

### 5. **Model Saving**
- Saves the trained model as `captcha_model.pth` for future use.

---

## Dataset Format

- All CAPTCHA images should be stored in a folder named `dataset`.
- Each image filename represents its label, e.g., `abc12.png` corresponds to the label "abc12".
- Images should ideally be of consistent format (e.g., `.png`, `.jpg`).
- Dataset Link : https://www.kaggle.com/datasets/hakantaskiner/captcha-dataset

---

## Setup and Usage

### Prerequisites

Install the required libraries:

pip install torch torchvision opencv-python scikit-learn tqdm
