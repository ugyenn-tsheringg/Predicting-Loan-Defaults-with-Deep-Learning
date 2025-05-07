# Predicting Loan Defaults with Deep Learning

A deep learning model to predict loan default probabilities for the African Credit Scoring Challenge.

<div align="center">
   <img src="results and visualizations/competition_ranking.jpeg">
</div>

## Motivation
This project was developed as part of a competitive challenge to address real-world financial risk assessment in African markets. It showcases my skills in deep learning, data preprocessing, and feature engineering while tackling a socially impactful problem. 

## Description

This project develops a robust neural network to predict the likelihood of loan defaults using financial data from the [African Credit Scoring Challenge](https://zindi.africa/competitions/african-credit-scoring-challenge). Built with TensorFlow, the model processes diverse features (e.g., loan amounts, durations, and borrower demographics) to provide accurate risk assessments for financial institutions in Africa's dynamic markets. Key techniques include feature engineering, SMOTE for class imbalance, and L2 regularization for robust generalization.

## Dataset
- Source: African Credit Scoring Challenge
- Features: Loan amounts, durations, country IDs, loan types, and more.
- Target: Binary (0: No default, 1: Default).

## Model Architecture
- Layers: 4 dense layers (256, 128, 64, 32 neurons) with ReLU activation.
- Regularization: L2 regularization and dropout (0.5, 0.4, 0.3).
- Optimization: Adam optimizer (learning rate: 0.0003).
- Callbacks: Early stopping and learning rate reduction on plateau.

## Features
- Predicts loan default probabilities using a deep neural network.
- Handles imbalanced data with SMOTE oversampling.
- Incorporates feature engineering (e.g., date-based features, log transformations).
- Visualizes data distributions, correlations, and model performance (e.g., ROC AUC, F1 Score).

## How It Works
## Exploratory Data Analysis (EDA)
- Inspected missing values, duplicates, and class distribution.
- Visualized data using histograms, boxplots, and heatmaps.
- Analyzed correlations to understand feature relationships with the target.

<div align="center">
   <img src="results and visualizations/eda_combined.png" width=700>
</div>

## Data Preprocessing:
- Log transformations for skewed features.
- Capped outliers at the 95th percentile to reduce extreme influence.
- Standard scaling and one-hot encoding for categorical variables.
- Feature engineering (e.g., loan duration in days, date-based features).

## Model Training:
- Stratified train-validation split to preserve class distribution.
- Applied SMOTE only to the training set to balance the minority class (defaults).
- Used early stopping and learning rate reduction callbacks for better convergence.
- Neural network trained with binary cross-entropy loss.

## Evaluation:
- Main metric: F1 Score (chosen due to class imbalance).
- Also reported: ROC AUC, Confusion Matrix, and Classification Report.

## Key Visualizations
- Distribution plots of raw and transformed features.
- Correlation heatmaps.
- Boxplots to detect outliers.
- Confusion matrix and loss curves during training.

## Performance
- F1 Score: 0.7168 (optimized with threshold tuning).
- ROC AUC: 0.9855, indicating strong discriminative ability.

Key Insight: Effective handling of class imbalance and feature skewness improved model robustness.

<div align="center">
   <img src="results and visualizations/loss_function.png" width=700>
</div>

## 🏆 Competition Results
- F1 Score (validation): 0.812
- Private Leaderboard: 0.6585
- Ranked: 335 out of 899 teams (Top 37%)

<div align="center">
   <img src="results and visualizations/submissions.jpeg" width=700>
</div>

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

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions.

License
MIT License (LICENSE)

