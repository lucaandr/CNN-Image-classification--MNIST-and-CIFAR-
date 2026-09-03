# Computer Vision & CNN Optimization Pipeline (MNIST & CIFAR-10)

An end-to-end Machine Learning study exploring the transition from simple grayscale digit recognition (**MNIST**) to complex RGB object classification (**CIFAR-10**). 

This project demonstrates automated hyperparameter tuning (**Optuna**, Grid/Random Search) and architectural engineering trade-offs using a modular **VGG-like CNN** with advanced regularization techniques.

---

## Key Performance Results

| Dataset | Architecture / Approach | Key Techniques | Epochs | Accuracy |
| :--- | :--- | :--- | :---: | :---: |
| **MNIST** | Custom CNN + Optuna | Bayesian Optimization, Softmax | 10 | **>98.0%** |
| **CIFAR-10** | Modular VGG-like CNN | Batch Normalization, Progressive Dropout | 12 | **~81.0%** |

---

## Strategic Shift: Automation vs. Architectural Design

A core engineering decision in this repository was recognizing hardware and compute constraints when scaling up dataset complexity:

* **MNIST Strategy (Automated Tuning):** Leveraged **Optuna (Bayesian Optimization)** to explore the hyperparameter space (learning rates, filter counts, dense layers). High accuracy was achieved with low compute costs.
* **CIFAR-10 Strategy (Engineering Pivot):** Exhaustive automated search on RGB images proved computationally prohibitive. The strategy pivoted to **manual architectural design based on VGG principles**, achieving **81% accuracy** in just 12 epochs without brute-force parameter searching.

---

## CIFAR-10 VGG-like Architecture

The final CIFAR-10 model utilizes a 3-block convolutional backbone designed to combat overfitting and ensure gradient stability:

* **Feature Extraction:** Hierarchical channels ($32 \rightarrow 64 \rightarrow 128$) using $3 \times 3$ kernels with `padding='same'`.
* **Gradient Stabilization:** `BatchNormalization()` after every convolutional layer to prevent internal covariate shift and allow higher learning rates ($lr=0.001$).
* **Progressive Regularization:** Escalating Dropout rate ($0.2 \rightarrow 0.3 \rightarrow 0.4 \rightarrow 0.5$) to prevent memorization in deeper, highly-parameterized layers.
* **Spatial Invariance:** `MaxPooling2D((2,2))` for downsampling and translation invariance.

---

## Project Structure

```text
.
├── image_clasv2.ipynb    # MNIST pipeline & Optuna hyperparameter tuning
├── cifar.ipynb           # CIFAR-10 VGG-like CNN training & real-time prediction
└── README.md             # Technical documentation
