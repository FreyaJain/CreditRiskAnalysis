# Loan Risk Prediction Project

This project develops a machine learning pipeline to predict loan risk using credit risk data. The pipeline handles data preprocessing, model training, hyperparameter optimization, and evaluation. It includes automated workflows for managing multiple classifiers and assessing performance through key metrics.

## Features
1. **Data Preprocessing:**
   - Removal of outliers based on domain knowledge.
   - Handling of missing values using iterative imputation.
   - Feature scaling and one-hot encoding with `ColumnTransformer`.

2. **Automated Model Optimization:**
   - Implementation of `Pipeline` and `RandomizedSearchCV` to automate hyperparameter tuning.
   - Supports multiple classifiers, including `RandomForestClassifier` and `LGBMClassifier`.

3. **Evaluation:**
   - Generation of precision-recall curves for model assessment.
   - Learning curve visualization to diagnose overfitting and underfitting.
   - Confusion matrices and classification reports for detailed insights.

4. **Model Refinement:**
   - Techniques to reduce overfitting: complexity reduction, feature selection, and regularization.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`

Install dependencies using:
```bash
pip install -r requirements.txt
# 
