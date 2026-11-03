# CIFAR-10 Neural Network Classification

## Overview

This project focuses on building and training neural networks to classify images from the CIFAR-10 dataset using PyTorch.

I started with a basic linear classifier, progressed to a Linear + Nonlinear fully connected neural network, and finally implemented a Convolutional Neural Network (CNN).

The objective is to understand how different architectures learn image features, compare their performance, and explore techniques to improve generalisation.

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 colour images belonging to 10 different classes.

- Training images: 50,000
- Testing images: 10,000
- Image dimensions: 32 × 32 × 3
- Number of classes: 10

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- Torchmetrics
- MLxtend

## Hardware Specifications

All experiments were conducted on the following hardware:

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 7445HS |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU VRAM | 6 GB |
| RAM | 16 GB |
| Framework | PyTorch |
| CUDA | 11.8 (PyTorch build) |

## Model Architectures and Experiments

I experimented with three different neural network architectures.

### 1. Linear Model

A basic fully connected linear classifier that served as the starting point for the experiments.

### 2. Linear + Nonlinear Model

A fully connected neural network incorporating multiple linear layers and ReLU activation functions. The ReLU activations introduce nonlinearity, allowing the network to learn more complex patterns.

### 3. Convolutional Neural Network (CNN)

The CNN uses convolutional layers, ReLU activation functions, and max-pooling layers to extract spatial features from images.

I experimented with:

- Different numbers of convolutional blocks, including two and three blocks.
- Different numbers of hidden units: 32, 64, 128 and 256.

After comparing the configurations, the two-convolution-block architecture with 64 hidden units gave the best fit among the configurations I explored.

The 128- and 256-hidden-unit configurations showed substantial overfitting, so I did not retain them as the selected architecture.

**Selected CNN configuration:**

- 2 convolutional blocks
- 64 hidden units
- ReLU activation
- Max pooling
- Fully connected output layer

## Results

The following table shows the progression across the three model architectures.

| Model | Training Accuracy | Testing Accuracy |
|---|---:|---:|
| Linear Model | 39.46% | 36.88% |
| Linear + Nonlinear Model | 48.30% | 45.85% |
| CNN (baseline) | 81% | 73.4% |

The results show a considerable improvement when moving from fully connected architectures to a CNN, which can learn spatial features from images.

*Accuracy figures represent the recorded experiments and their respective configurations.*

## CPU vs GPU Performance

I also compared training times across CPU and GPU to understand how different architectures benefit from GPU acceleration.

| Model | CPU Training Time | GPU Training Time |
|---|---:|---:|
| Linear Model | 98.84 s | 87.93 s |
| Linear + Nonlinear Model | 91.53 s | 90.60 s |
| CNN (32 hidden units) | 323.67 s | 103.43 s |
| CNN (64 hidden units) | 791.58 s | 104.63 s |

*Times are recorded training times from the notebook runs, in seconds. They are environment- and run-dependent and should be treated as indicative rather than formal benchmark measurements.*

The comparison highlights how GPU acceleration becomes increasingly beneficial for computationally intensive CNN architectures.

## Training and Evaluation

I implemented reusable training and testing functions in PyTorch to keep the experiments organised.

The models were evaluated using:

- Training loss
- Testing loss
- Training accuracy
- Testing accuracy
- Confusion matrix
- Sample predictions

## Project Progress

| Date | Milestone |
|---|---|
| 3 November 2026 | Completed the basic CNN architecture, experimented with different convolutional block configurations and hidden-unit sizes, and established the baseline. |
| Upcoming | Apply data augmentation and evaluate its effect on model performance. |

## Future Improvements

- Apply data augmentation to improve generalisation.
- Experiment with Batch Normalization and Dropout.
- Explore different optimizers and learning-rate schedulers.
- Implement adversarial attacks such as FGSM.
- Explore adversarial training and other approaches to improve model robustness.

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Open the Jupyter Notebook.
4. Run the cells sequentially to reproduce the experiments.
