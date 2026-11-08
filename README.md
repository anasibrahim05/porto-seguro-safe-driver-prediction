# Porto Seguro Safe Driver Prediction

A machine learning and deep learning project exploring imbalanced binary classification using the Porto Seguro Safe Driver Prediction dataset from Kaggle.

## Project Overview

The objective of this project is to predict the probability that a driver will initiate an insurance claim in the following year.

This project explores different machine learning and deep learning approaches, comparing their performance and investigating how feature representation and neural network architectures influence predictive performance.

The experiments progress from classical machine learning baselines to PyTorch-based neural networks, including categorical embeddings and hyperparameter optimization with Optuna.

## Dataset

**Source:** [Kaggle - Porto Seguro Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction)

The dataset contains anonymized driver-related features and a binary target variable.

- `target = 0`: No insurance claim.
- `target = 1`: Insurance claim.

The dataset is highly imbalanced, making appropriate evaluation metrics essential.

**Note:** Download the competition data directly from Kaggle. The original dataset files are not included in this repository.

## Project Workflow

### 1. Exploratory Data Analysis
- Dataset structure and feature analysis.
- Target distribution and class imbalance.
- Missing and unknown value analysis.
- Numerical and categorical feature exploration.

### 2. Data Preprocessing
- Numerical missing-value imputation using the median.
- Separate categorical representation for missing values.
- Missing-value indicator features.
- Feature grouping into categorical, numerical, and binary columns.
- Train, validation, and test split.
- Numerical standardization and categorical encoding.

Preprocessing transformations were fitted using training data to avoid data leakage.

### 3. Classical Machine Learning

Two classical algorithms were implemented as baseline models:

- Logistic Regression with class weighting.
- Random Forest with balanced subsampling.

### 4. Deep Learning with PyTorch

**One-Hot Encoding MLP**

A Multi-Layer Perceptron using one-hot encoded categorical features, with:
- Fully connected layers.
- Batch Normalization.
- ReLU activation.
- Dropout regularization.
- BCEWithLogitsLoss with positive class weighting.
- Adam optimizer.

**Hyperparameter Optimization with Optuna**

Optuna was used to search for suitable MLP configurations, including:
- Hidden layer dimensions.
- Dropout rates.
- Learning rate.
- Weight decay.
- Batch size.
- Optimizer selection.

The optimization objective was validation PR-AUC.

**Categorical Embeddings MLP**

An alternative MLP architecture using trainable embedding layers for categorical features.

The learned categorical representations were combined with numerical and binary features before passing them through fully connected layers.

## Evaluation Metrics

Because the dataset is highly imbalanced, accuracy is not used as the primary evaluation metric.

The project uses:

- **ROC-AUC:** Measures the model's ability to rank positive examples above negative examples.
- **Normalized Gini:** The competition's evaluation metric.
- **PR-AUC (Average Precision):** Evaluates precision and recall, particularly useful for imbalanced classification.

The normalized Gini coefficient is calculated from ROC-AUC:

\[
Gini = 2 \times ROC\text{-}AUC - 1
\]

## Results

The following table summarizes the best observed validation metrics for each model.

| Model | ROC-AUC | Normalized Gini | PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.6342 | 0.2684 | 0.06783 |
| Random Forest | 0.6256 | 0.2513 | 0.06110 |
| One-Hot MLP | **0.6425** | **0.2850** | 0.06995 |
| Embedding MLP | 0.6367 | 0.2734 | **0.07209** |

**Key observations:**
- One-Hot MLP achieved the highest ROC-AUC and normalized Gini.
- Embedding MLP achieved the highest PR-AUC.
- Logistic Regression provided a competitive baseline.
- Random Forest did not outperform Logistic Regression with the selected configuration.

*Note: The best ROC-AUC and PR-AUC values for a model may correspond to different training epochs. These are validation results, not official Kaggle leaderboard scores. The Embedding MLP result comes from an unseeded exploratory run.*

## Repository Structure

```text
porto-seguro-safe-driver-prediction/
│
├── notebooks/
│   └── porto_seguro_analysis.ipynb
│
├── reports/
│   └── figures/
│
├── data/
│   └── README.md
│
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/porto-seguro-safe-driver-prediction.git

cd porto-seguro-safe-driver-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Download the dataset

Download the competition dataset from Kaggle and place the required CSV files in the appropriate data directory.

Refer to `data/README.md` for dataset placement instructions.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `notebooks/porto_seguro_analysis.ipynb` and execute the cells.

## Key Learnings

Through this project, I explored:

- Handling missing values and categorical features.
- Building preprocessing workflows while avoiding data leakage.
- Classical machine learning baselines.
- Neural network development using PyTorch.
- Categorical embeddings versus one-hot encoding.
- Handling class imbalance with weighted loss functions.
- Hyperparameter optimization using Optuna.
- Batch Normalization, Dropout, and early stopping.
- ROC-AUC, PR-AUC, and normalized Gini evaluation.
- Model comparison and validation-based experimentation.

## Conclusion

This project provided practical experience in developing and evaluating machine learning and deep learning models for an imbalanced classification problem.

The experiments demonstrated that model complexity and feature representation do not automatically guarantee better performance. The most suitable approach depends on the evaluation metric and experimental results.

The primary goal was to develop a deeper understanding of the end-to-end machine learning experimentation workflow.

## Author

**Anas Ibrahim**

BS Computer Science Student | Machine Learning & Deep Learning Enthusiast

## Acknowledgements

- [Porto Seguro](https://www.portoseguro.com.br/)
- [Kaggle Competition](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction)
- [PyTorch](https://pytorch.org/)
- [Optuna](https://optuna.org/)
