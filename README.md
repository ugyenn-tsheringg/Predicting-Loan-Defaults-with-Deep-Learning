# Predicting Loan Defaults with Deep Learning

A deep learning model to predict loan default probabilities for the African Credit Scoring Challenge.

## Description

This project develops a robust neural network to predict the likelihood of loan defaults using financial data from the [African Credit Scoring Challenge](https://zindi.africa/competitions/african-credit-scoring-challenge). Built with TensorFlow, the model processes diverse features (e.g., loan amounts, durations, and borrower demographics) to provide accurate risk assessments for financial institutions in Africa's dynamic markets. Key techniques include feature engineering, SMOTE for class imbalance, and L2 regularization for robust generalization.

## Features

- Predicts loan default probabilities using a deep neural network.
- Handles imbalanced data with SMOTE oversampling.
- Incorporates feature engineering (e.g., date-based features, log transformations).
- Visualizes data distributions, correlations, and model performance (e.g., ROC AUC, F1 Score).

## Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn, TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **Data**: African Credit Scoring Challenge dataset
- **Environment**: Google Colab, Kaggle

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/your-username/your-repo.git
```
2. Install dependencies:
  ```bash
    pip install -r requirements.txt
  ```

## Usage
Run the main script to train the model and generate predictions:
```bash
python predicting_loan_defaults_with_deep_learning.py
```
Outputs:
submission.csv: Test set predictions.

Visualizations: Data distributions, correlation heatmaps, and confusion matrix.

## Model Architecture
- Layers: 4 dense layers (256, 128, 64, 32 neurons) with ReLU activation.
- Regularization: L2 regularization and dropout (0.5, 0.4, 0.3).
- Optimization: Adam optimizer (learning rate: 0.0003).
- Callbacks: Early stopping and learning rate reduction on plateau.

## Performance
- F1 Score: ~0.XX (optimized with threshold tuning).
- ROC AUC: ~0.XX, indicating strong discriminative ability.

Key Insight: Effective handling of class imbalance and feature skewness improved model robustness.

## Dataset
Source: African Credit Scoring Challenge

Features: Loan amounts, durations, country IDs, loan types, and more.

Target: Binary (0: No default, 1: Default).

## How It Works
## Data Preprocessing:
- Log transformations for skewed features.
- Standard scaling and one-hot encoding for categorical variables.
- Feature engineering (e.g., loan duration in days, date-based features).

## Model Training:
- SMOTE to balance the minority class (defaults).
- Neural network trained with binary cross-entropy loss.

## Evaluation:
- Threshold tuning to optimize F1 Score.
- Visualizations for model performance and data insights.

## Motivation
This project was developed as part of a competitive challenge to address real-world financial risk assessment in African markets. It showcases my skills in deep learning, data preprocessing, and feature engineering while tackling a socially impactful problem.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions.

License
MIT License (LICENSE)

