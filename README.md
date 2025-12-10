# Telco Customer Churn — Classification Project

A complete machine learning classification project that predicts customer churn using **Logistic Regression** and **Random Forest** models. The project is organized into three modular notebooks for preprocessing, training, and visualization.


## Notebooks

### 1. Preprocessing (`1_Preprocessing.ipynb`)
- Load and inspect the dataset
- Handle missing values and data type conversions
- Encode categorical variables and target
- Build preprocessing pipelines (StandardScaler, OneHotEncoder)
- Split data (80/20 train/test with stratification)
- Save preprocessed data and pipelines

**Outputs:** `preprocessed_data.pkl`, `preprocessor.pkl`

### 2. Model Training (`2_Model_Training.ipynb`)
- Load preprocessed data
- Train Logistic Regression classifier
- Train Random Forest classifier (200 estimators)
- Evaluate both models using:
  - Accuracy
  - ROC AUC
  - Confusion Matrix
  - Classification Report
  - MAE / RMSE
- Compare models and recommend the best
- Save trained pipelines

**Outputs:** `lr_model.joblib`, `rf_model.joblib`, `best_model.joblib`, `model_metrics.pkl`

### 3. Visualization (`3_Visualization.ipynb`)
- Load trained models and test data
- Plot confusion matrices (side-by-side comparison)
- Visualize predicted probabilities vs actual labels
- Display top 20 feature importances (Random Forest)
- Compare model metrics (bar chart)
- Predict on new sample
- Generate summary report

**Outputs:** PNG charts (confusion_matrices.png, predicted_vs_actual.png, feature_importances.png, model_comparison.png)


