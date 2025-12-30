# Deep Feed-Forward Neural Network for Mushroom Classification

This repository presents a supervised learning project that applies a **deep feed-forward neural network (FFNN)** to a real-world–style classification task and rigorously compares its performance against a simpler, traditional machine learning model. The goal is to evaluate **when deep learning is warranted** and when a simpler model may fall short.

The task is **binary classification**: predicting whether a mushroom is **edible** or **poisonous** based on physical and environmental characteristics.

While the .ipynb file is available, access through Google Colab is recommended. Link: [https://colab.research.google.com/drive/1w3274Xwo8C8md8iPraaK4ONNZUBMtfJd?usp=sharing]

This project uses the mushroom dataset found here: [https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset]

---

## Project Overview

- **Problem type:** Binary classification  
- **Target variable:** Mushroom class (`edible` vs. `poisonous`)
- **Dataset characteristics:**
  - Mix of categorical and continuous features
  - High-dimensional after encoding
  - Fairly balanced target classes (~54% poisonous, ~46% edible)

This project emphasizes:
- Careful **data cleaning and preprocessing**
- Thoughtful **neural network architecture design**
- Use of **regularization** to prevent overfitting
- Transparent **training and validation evaluation**
- A direct comparison with a **simpler baseline model** to justify model complexity

---

## Data Preparation

Before modeling, the dataset undergoes the following steps:

- Columns with excessive missing values are removed
- Remaining rows with missing data are dropped
- Target variable is encoded numerically:
  - `edible = 1`
  - `poisonous = 0`
- **Categorical features** are one-hot encoded
- **Continuous features** (e.g., cap diameter, stem height) are standardized using **Z-score scaling**
- Final dataset is split into **80% training / 20% testing**

These steps ensure compatibility with both neural networks and traditional machine learning models.

---

## Deep Learning Model (Primary Model)

A **Feed-Forward Neural Network** is built using **TensorFlow / Keras**.

### Architecture

- **Input layer:** All encoded and scaled features
- **Hidden Layer 1:**  
  - 128 neurons  
  - ReLU activation  
  - Dropout (0.3) for regularization
- **Hidden Layer 2:**  
  - 64 neurons  
  - ReLU activation  
  - Dropout (0.3)
- **Hidden Layer 3:**  
  - 32 neurons  
  - ReLU activation
- **Output Layer:**  
  - 1 neuron  
  - Sigmoid activation (binary classification)

### Training Configuration

- **Loss function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Batch size:** 32  
- **Epochs:** 50  
- **Evaluation metrics:** Accuracy (tracked for both training and validation)

### Performance

- Training Accuracy: **~99.87%**
- Test Accuracy: **~99.80%**
- Training and validation loss converge rapidly with minimal gap, indicating strong generalization and limited overfitting.

---

## Baseline Model (Comparison)

To assess whether deep learning is necessary, a **Logistic Regression** model is trained on the **same preprocessed data** and **same train/test split**.

- Assumes linear decision boundaries
- Uses sigmoid output for probability estimation
- No hidden layers or nonlinear feature interactions

### Baseline Performance

- Accuracy: **~79%**

This substantial performance gap demonstrates that the relationships between features and mushroom edibility are **nonlinear** and benefit significantly from the representational power of a deep neural network.

---

## Key Findings

- The deep neural network **outperforms Logistic Regression by over 20 percentage points**
- Regularization via **Dropout** effectively prevents overfitting
- Neural networks are particularly well-suited for datasets with:
  - Many categorical variables
  - Complex feature interactions
  - Nonlinear decision boundaries

The results clearly justify the use of a deep learning model for this task.

---

## Repository Contents

- **`MLFA25_Mushroom.ipynb`**  
  End-to-end Google Colab notebook including:
  - Data cleaning and preprocessing
  - Feature encoding and scaling
  - Neural network construction and training
  - Baseline model training
  - Evaluation and visualizations of loss/accuracy

- **`MLFA25 - Mushroom Report.pdf`**  
  Technical report detailing:
  - Dataset analysis
  - Model architecture and design choices
  - Training behavior and results
  - Comparative analysis and conclusions :contentReference[oaicite:0]{index=0}

---

## Running the Project (Google Colab)

This project is designed to be run in **Google Colab**.

After uploading the repository to GitHub, open the notebook using: https://colab.research.google.com/drive/1w3274Xwo8C8md8iPraaK4ONNZUBMtfJd?usp=sharing

