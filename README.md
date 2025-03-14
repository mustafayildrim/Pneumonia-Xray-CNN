# Deep Learning for Pneumonia Detection: Classifying Chest X-rays with CNNs in PyTorch

## Overview
This project utilizes a **Convolutional Neural Network (CNN)** implemented in **PyTorch** to classify chest X-ray images as either **normal** or indicating **pneumonia**. The goal is to develop an AI-assisted diagnostic tool that can help identify pneumonia from medical imaging.

## Dataset
The dataset used in this project is the **Chest X-ray Pneumonia Dataset** from Kaggle:
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Dataset Structure:
- **Training Set**: Labeled X-ray images for training the model.
- **Validation Set**: Images for fine-tuning hyperparameters.
- **Test Set**: Unseen images to evaluate model performance.

## Model
- The model is a **CNN (Convolutional Neural Network)** built with **PyTorch**.
- Uses convolutional layers to extract features from X-ray images.
- Includes **batch normalization** and **dropout** for regularization.
- Applies **softmax activation** for classification.

## Installation
### Prerequisites
Ensure you have Python and the following libraries installed:
```bash
pip install torch torchvision numpy matplotlib opencv-python pandas
```

## Usage
1. **Load the dataset**: Ensure the dataset is available in the specified directory.
2. **Train the model**: Run the training script to train the CNN.
3. **Evaluate performance**: Use the test dataset to analyze accuracy and loss.

## Results
- The model is evaluated using accuracy, precision, recall, and F1-score.
- Visualizations of training loss and accuracy trends are included.
- Example X-ray images with predicted labels are displayed.

## Future Improvements
- Fine-tuning hyperparameters for better performance.
- Exploring different CNN architectures such as ResNet.
- Expanding the dataset for more diverse cases.


