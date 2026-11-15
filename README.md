# CIFAR-10 Neural Network Classification & Adversarial Robustness

## Overview

This project explores image classification on the CIFAR-10 dataset using PyTorch, progressing from basic fully connected neural networks to Convolutional Neural Networks (CNNs).

The project initially focused on understanding how different architectures learn image features and how techniques such as data augmentation and regularisation can improve generalisation.

It was then extended to investigate **adversarial robustness** by implementing two gradient-based adversarial attacks from scratch:

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)

The objective is to evaluate how sensitive the trained CNN is to small, deliberately crafted perturbations and compare the effectiveness of these two attacks.

## Dataset

The project uses the CIFAR-10 dataset, consisting of 60,000 colour images across 10 classes.

| Property | Description |
|---|---|
| Training images | 50,000 |
| Testing images | 10,000 |
| Image dimensions | 32 × 32 × 3 |
| Number of classes | 10 |
| Classes | Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck |

The dataset was loaded using `torchvision.datasets.CIFAR10`.

Images were converted to floating-point tensors, scaled to [0, 1], and normalised using the CIFAR-10 channel-wise mean and standard deviation.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Torchmetrics
- Matplotlib
- Pandas
- MLxtend
- Jupyter Notebook

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

A basic fully connected linear classifier that served as the starting point for understanding image classification.

### 2. Linear + Nonlinear Model

A fully connected neural network incorporating multiple linear layers and ReLU activation functions. The nonlinear activations allow the network to learn more complex relationships between input pixels and image classes.

### 3. Convolutional Neural Network (CNN)

The CNN uses convolutional layers to extract spatial features, ReLU activation functions to introduce nonlinearity, and max-pooling layers to reduce spatial dimensions.

I experimented with:

- Different numbers of convolutional blocks, including two and three blocks.
- Different hidden-unit configurations: 32, 64, 128 and 256.
- Batch Normalization and Dropout.
- Different optimisers and learning-rate scheduling strategies.

The initial CNN experiments established a substantial improvement over the fully connected architectures. Subsequent experiments incorporated data augmentation and regularisation to improve generalisation.

## Data Augmentation and Regularisation

To improve the model's ability to generalise to unseen images, I experimented with several data augmentation techniques.

These included:

- RandomHorizontalFlip
- RandomCrop
- RandomErasing
- ColorJitter
- TrivialAugmentWide
- MixUp
- CutMix

I also explored Batch Normalization and Dropout to help control overfitting.

Not every augmentation improved performance. Some combinations reduced training accuracy or failed to improve test accuracy. These experiments helped identify the importance of evaluating augmentation strategies rather than assuming that more augmentation necessarily leads to better results.

## Classification Results

The initial architecture experiments showed the following progression:

| Model | Training Accuracy | Testing Accuracy |
|---|---:|---:|
| Linear Model | 39.46% | 36.88% |
| Linear + Nonlinear Model | 48.30% | 45.85% |
| Initial CNN | 81% | 73.4% |

After further experimentation with CNN architectures, augmentation and regularisation, a later model achieved the following clean test performance:

| Metric | Result |
|---|---:|
| Clean Test Accuracy | **89.72%** |
| Clean Test Loss | **0.3102** |

These figures represent different stages of experimentation: the initial CNN result is not the final model's clean test result.

## CPU vs GPU Performance

I also compared training times across CPU and GPU to understand how different architectures benefit from GPU acceleration. One thing to note is that the CNN 64 units time for GPU is actually the time taken by it for running 128 hidden units. I had found 128 units to be the best number, but it was taking very very long to execute on the CPU, so I changed it to 64 units for CPU 

| Model | CPU Training Time | GPU Training Time |
|---|---:|---:|
| Linear Model | 98.84 s | 87.93 s |
| Linear + Nonlinear Model | 91.53 s | 90.60 s |
| CNN (32 hidden units) | 323.67 s | 103.43 s |
| CNN  |(64 hidden units) 791.58 s | (128 hidden units) 104.63 s |

Training times are taken from individual notebook runs and are environment- and run-dependent. They should be treated as indicative rather than formal benchmark measurements.

## Adversarial Robustness Experiments

After establishing a CNN classifier, I extended the project to investigate adversarial attacks.

The experiments focus on generating adversarial examples that cause a trained model to misclassify images through carefully calculated perturbations.

Both attacks were implemented manually using PyTorch and automatic differentiation.

### 1. Fast Gradient Sign Method (FGSM)

FGSM is a single-step gradient-based attack. It uses the sign of the gradient with respect to the input image to determine the direction in which the input should be perturbed. It makes changes to the input, that is the image rather than the weights and bias

The update is:

$x_{adv} = \text{clip}(x + \epsilon \cdot \text{sign}(\nabla_x J(\theta,x,y)), 0, 1)$

Where:

- \(x\) is the original image.
- \(\epsilon\) controls the perturbation magnitude.
- \(J\) is the loss function.
- \(\nabla_x J\) is the gradient of the loss with respect to the input.

The attack was implemented in the original pixel space, with images denormalised before perturbation and normalised again before model evaluation.

**FGSM results:**

| Epsilon | Test Loss | Test Accuracy |
|---:|---:|---:|
| 0 | 0.3102 | 89.72% |
| 0.001 | 0.5386 | 82.86% |
| 0.005 | 2.3313 | 45.63% |
| 0.01 | 5.2885 | 18.48% |
| 0.05 | 9.8982 | 2.43% |

The results show a substantial reduction in classification accuracy as the perturbation magnitude increases. The graph for loss vs epsilon and accuracy vs epsilon is in Plot_images/FGSM

### 2. Projected Gradient Descent (PGD)

PGD is an iterative gradient-based attack. It is basically an iterative FGSM. Instead of applying a single perturbation, it repeatedly updates the adversarial image using small gradient steps. This makes it much more reliable and powerful than FGSM, as instead of one big perturbation, it at each iteration calculates the gradient and moves such that it stays withing the epsilon ball area. So it prevents from overshooting

The attack also projects the perturbed image back into the permitted epsilon neighbourhood of the original image after each update.

The implementation uses:

- Random initialisation within the permitted perturbation range.
- Iterative gradient-based updates.
- Projection onto the \(L_\infty\) perturbation constraint.
- Clipping to maintain valid pixel values in the [0, 1] range.

For the epsilon comparison experiments, I used 10 iterations and set the step size to:

\[
\alpha = \frac{\epsilon}{4}
\]

**PGD results:**

| Epsilon | Test Loss | Test Accuracy |
|---:|---:|---:|
| 0 | 0.3102 | 89.72% |
| 0.001 | 0.5464 | 82.69% |
| 0.005 | 3.0046 | 36.95% |
| 0.01 | 8.7593 | 6.78% |
| 0.05 | 43.5788 | 0.00% |

The graphs for loss vs epsilon and accuracy vs epsilon for PGD can be found at Plot_images/PGD

### 3. FGSM vs PGD Comparison

I compared both attacks using the same epsilon values to understand their relative effectiveness.

| Epsilon | FGSM Accuracy | PGD Accuracy |
|---:|---:|---:|
| 0 | 89.72% | 89.72% |
| 0.001 | 82.86% | 82.69% |
| 0.005 | 45.63% | 36.95% |
| 0.01 | 18.48% | 6.78% |
| 0.05 | 2.43% | 0.00% |

Across the tested non-zero epsilon values, PGD achieved lower classification accuracy than FGSM. This indicates that PGD was the stronger attack under the configurations tested.

The comparison also demonstrates how iterative attacks can produce stronger adversarial effects than a single-step attack.

### 4. Visualising Adversarial Examples

I also visualised the original image alongside its FGSM- and PGD-perturbed versions to observe how adversarial perturbations affect the model's predictions.

The project includes plots showing:

- FGSM accuracy vs epsilon.
- FGSM loss vs epsilon.
- PGD accuracy vs epsilon.
- PGD loss vs epsilon.
- FGSM vs PGD accuracy comparison.
- FGSM vs PGD loss comparison.

The plots are available in the `Plot_images` directory.

## Project Structure

```text
CIFAR-10_Classification/
│
├── CIFAR-10_Neural_Network_Classification.ipynb
│
├── Plot_images/
│   ├── FGSM/
│   │   ├── FGSM_Accuracy_vs_Epsilon.png
│   │   └── FGSM_loss_vs_Epsilon.png
│   │
│   ├── PGD/
│   │   ├── PGD_Accuracy_vs_Epsilon.png
│   │   └── PGD_loss_vs_Epsilon.png
│   │
│   └── Comparison_PGD_FGSM/
│       ├── PGD_Accuracy_vs_FGSM_Accuracy.png
│       └── PGD_loss_vs_FGSM_loss.png
│
├── models/
│
├── data/
│
└── README.md
```

The `data` and `models` directories are used locally for dataset and model-related files; they are not necessarily included in the GitHub repository.

## Training and Evaluation

I implemented reusable training and testing functions in PyTorch to keep the experiments organised.

The models were evaluated using:

- Training and testing loss.
- Training and testing accuracy.
- Confusion matrices.
- Sample predictions.
- Adversarial test accuracy and loss.
- Accuracy and loss comparisons across perturbation magnitudes.

## Key Observations

- CNNs substantially outperformed the initial fully connected architectures on CIFAR-10.
- Data augmentation and regularisation provided additional opportunities to improve generalisation, although their effectiveness varied across experiments.
- Both FGSM and PGD caused significant reductions in classification accuracy as epsilon increased.
- PGD produced a stronger reduction in accuracy than FGSM across the tested non-zero epsilon values.
- At epsilon = 0.01, PGD reduced accuracy to 6.78%, compared with 18.48% for FGSM.
- At epsilon = 0.05, PGD reduced the measured accuracy to 0%.

## Future Improvements

- Investigate adversarial training to improve model robustness.
- Study the effect of PGD step size and number of iterations on attack effectiveness.
- Evaluate class-wise vulnerability to adversarial perturbations.
- Explore additional adversarial attack methods.
- Investigate the trade-off between clean accuracy and adversarial robustness.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/PranavAgrawal07/CIFAR-10-Neural-Network-Classification.git
   ```

2. Navigate to the project directory:

   ```bash
   cd CIFAR-10-Neural-Network-Classification
   ```

3. Install the required dependencies:

   ```bash
   pip install torch torchvision torchmetrics matplotlib pandas mlxtend jupyter
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `CIFAR-10_Neural_Network_Classification.ipynb` and execute the cells sequentially.

The notebook downloads CIFAR-10 through Torchvision when the dataset is not already available locally.

---

**Note:** The reported results correspond to the configurations and notebook runs documented in this project. Training times and adversarial attack measurements may vary with hardware, random initialisation and experimental settings.
