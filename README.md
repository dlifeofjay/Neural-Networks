# Neural Networks from Scratch

A collection of neural network implementations built entirely from first principles using **NumPy**, without relying on high-level deep learning frameworks such as TensorFlow or PyTorch. These projects demonstrate a deep understanding of the mathematical foundations underpinning modern machine learning — including forward propagation, backpropagation, gradient descent, and activation functions.

---

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [1. Handwritten Digit Classifier](#1-handwritten-digit-classifier)
  - [2. Retail Sales Analysis & Forecasting](#2-retail-sales-analysis--forecasting)
- [Key Concepts](#key-concepts)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Results](#results)
- [License](#license)

---

## Overview

This repository contains two end-to-end machine learning projects that showcase the design, training, and evaluation of neural networks implemented **from scratch**. The goal is to move beyond black-box usage of ML libraries and build a thorough, hands-on understanding of:

- How neural networks learn through iterative weight updates.
- The role of activation functions (ReLU, Softmax) in introducing non-linearity.
- The mechanics of backpropagation and gradient-based optimisation.
- Practical data preprocessing, exploration, and feature engineering workflows.

---

## Projects

### 1. Handwritten Digit Classifier

**Notebook:** `Neural Network.ipynb`

A fully connected, two-layer neural network trained on the [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) dataset (a variant of the classic MNIST benchmark) to classify 28×28 grayscale images of handwritten digits (0–9).

#### Architecture

| Layer | Dimensions | Activation |
|-------|-----------|------------|
| Input | 784 features (28×28 pixels) | — |
| Hidden | 10 neurons | ReLU |
| Output | 10 neurons (one per class) | Softmax |

#### Implementation Highlights

- **Data Pipeline** — Raw CSV data is loaded, converted to NumPy arrays, shuffled for randomness, and split into training (41,000 samples) and test (1,000 samples) sets. Feature matrices are transposed for efficient matrix multiplication.
- **Parameter Initialisation** — Weights are drawn from a standard normal distribution and scaled by 0.01 to break symmetry; biases are zero-initialised.
- **Forward Pass** — Implements the canonical `Z = WX + b` → activation pipeline across both layers.
- **Loss & Backpropagation** — Computes the cross-entropy loss gradient via the difference between Softmax predictions and one-hot encoded labels, then propagates error signals backward through each layer using the chain rule.
- **Optimiser** — Vanilla gradient descent with a configurable learning rate (`α = 0.001`).
- **Evaluation** — Accuracy is printed at every epoch; individual predictions are visualised alongside their ground-truth labels using Matplotlib.

#### Performance

The model achieves **~84% accuracy** on the held-out test set after 100 training epochs — a strong result for a simple two-layer network with no regularisation, data augmentation, or advanced optimisers.

---

### 2. Retail Sales Analysis & Forecasting

**Notebook:** `Project2.ipynb`

An end-to-end data science workflow that explores a synthetic retail sales dataset, performs cleaning and feature engineering, and applies a neural network to forecast sales across multiple product categories and regions.

#### Dataset

The `retail_sales.csv` dataset contains **1,825 daily records** spanning the full year of 2023, with the following features:

| Feature | Description |
|---------|-------------|
| Date | Transaction date |
| Category | Product category (Electronics, Clothing, Home Goods, Sports, Books) |
| Sales | Revenue amount |
| Quantity | Units sold |
| Profit | Profit generated |
| Region | Geographic region (North, South, East, West) |

#### Workflow

1. **Exploratory Data Analysis** — Inspects data types, distributions, and summary statistics using Pandas and Seaborn.
2. **Data Cleaning** — Identifies and handles missing values (NaN entries in Category, Sales, Quantity, and Region columns); removes corrupt "NaN?" category labels; resets the dataframe index.
3. **Feature Engineering** — Pivots the cleaned data into a time-series format with one column per product category, sorted by date. Remaining NaN values are imputed with column medians.
4. **Visualisation** — Generates time-series plots of daily sales by category to identify trends, seasonality, and anomalies.
5. **Neural Network Forecasting** — Trains a neural network model on the engineered features to predict future sales values.

---

## Key Concepts

This repository provides practical, code-level demonstrations of the following machine learning fundamentals:

- **ReLU Activation** — Introduces non-linearity by zeroing out negative values: `f(z) = max(0, z)`.
- **Softmax Activation** — Converts raw logits into a probability distribution over classes.
- **One-Hot Encoding** — Transforms integer class labels into binary indicator vectors for loss computation.
- **Backpropagation** — Applies the chain rule to compute the gradient of the loss function with respect to every weight and bias in the network.
- **Gradient Descent** — Iteratively adjusts parameters in the direction that minimises the loss.
- **Data Preprocessing** — Covers shuffling, splitting, normalisation, missing-value imputation, and pivot-based feature engineering.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Python 3.11+](https://www.python.org/) | Programming language |
| [NumPy](https://numpy.org/) | Numerical computation and linear algebra |
| [Pandas](https://pandas.pydata.org/) | Data manipulation and analysis |
| [Matplotlib](https://matplotlib.org/) | Plotting and visualisation |
| [Seaborn](https://seaborn.pydata.org/) | Statistical data visualisation |
| [Jupyter Notebook](https://jupyter.org/) | Interactive development environment |

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Neural Networks"

# Install dependencies
pip install numpy pandas matplotlib seaborn jupyter
```

### Running the Notebooks

```bash
jupyter notebook
```

Then open either `Neural Network.ipynb` or `Project2.ipynb` from the Jupyter interface.

> **Note:** The Digit Recognizer notebook expects the training data at a specific path. Update the `pd.read_csv(...)` call in the first code cell to point to your local copy of the [Kaggle Digit Recognizer `train.csv`](https://www.kaggle.com/c/digit-recognizer/data). Similarly, `Project2.ipynb` expects `retail_sales.csv` to be in the working directory.

---

## Results

| Project | Metric | Value |
|---------|--------|-------|
| Digit Classifier | Test Accuracy | **~84%** (100 epochs, α=0.001) |
| Sales Forecasting | — | Time-series analysis and prediction across 5 product categories |

---

## License

This project is available for educational and portfolio purposes. Feel free to explore, learn from, and build upon this work.
