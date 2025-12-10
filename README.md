# Telco Customer Churn — Classification Project

A complete machine learning classification project that predicts customer churn using **Logistic Regression** and **Random Forest** models. The project is organized into three modular notebooks for preprocessing, training, and visualization.

## Project Structure

```
e:\BI project\Bi class\
├── 1_Preprocessing.ipynb          # Data cleaning & feature engineering
├── 2_Model_Training.ipynb         # Train & evaluate both models
├── 3_Visualization.ipynb          # Visualize results & predictions
├── Telco-Customer-Churn.csv       # Input dataset
├── requirements.txt               # Package dependencies
└── README.md                      # This file
```

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

## Quick Start

1. **Install dependencies:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Run notebooks in order:**
```powershell
jupyter notebook
```
   - Open `1_Preprocessing.ipynb` and run all cells
   - Open `2_Model_Training.ipynb` and run all cells
   - Open `3_Visualization.ipynb` and run all cells

## Results Summary

**Model Performance (on test set):**
- **Logistic Regression:** Accuracy ~80%, ROC AUC ~0.84
- **Random Forest:** Accuracy ~77%, ROC AUC ~0.81

**Recommended Model:** Logistic Regression (better accuracy & interpretability)

**Top Features for Churn:**
1. TotalCharges
2. tenure
3. MonthlyCharges
4. Contract type (Month-to-month)
5. Payment method (Electronic check)

## Files Generated

After running all notebooks, you'll have:
- **Models:** `lr_model.joblib`, `rf_model.joblib`, `best_model.joblib`
- **Data:** `preprocessed_data.pkl`, `preprocessor.pkl`, `model_metrics.pkl`
- **Charts:** `confusion_matrices.png`, `predicted_vs_actual.png`, `feature_importances.png`, `model_comparison.png`

## Predict on New Data

To predict churn for a new customer:

```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('best_model.joblib')

# Create customer data
customer = pd.DataFrame([{
    'gender': 'Female',
    'tenure': 12,
    'MonthlyCharges': 70.35,
    'Contract': 'Month-to-month',
    # ... add all required features
}])

# Predict
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0, 1]
print(f"Churn prediction: {prediction}, Probability: {probability:.4f}")
```

## Next Steps (Optional)

- **Hyperparameter Tuning:** Use GridSearchCV or RandomizedSearchCV
- **Cross-Validation:** Implement k-fold cross-validation for more robust metrics
- **Class Imbalance:** Try SMOTE or class weights if churn is severely imbalanced
- **Deployment:** Save best model and create a simple Flask API for predictions
