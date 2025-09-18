# Telco Customer Churn Prediction

## 1️⃣ Problem Statement
Predict which telecom customers are likely to churn (cancel service) to help business take targeted retention actions.

## 2️⃣ Dataset
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Size: 7043 rows × 21 columns
- Key Columns: 
  - `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn` (target)

## 3️⃣ EDA Highlights
- Churn is concentrated in **Month-to-Month** contract customers.
- Customers with **high MonthlyCharges** more likely to churn.
- **Tenure** is usually lower for churners.
- Binary features (`Yes`/`No`) converted to 0/1.
- Added **average charges per month** and **tenure groups** for feature engineering.

### Example Plots:
- Churn distribution  
- Churn by Contract  
- Tenure vs Churn  
- MonthlyCharges distribution  

## 4️⃣ Modeling
- Train/Test split: 80/20 (stratified)
- Preprocessing: ColumnTransformer
  - Numerical → median imputer + StandardScaler
  - Categorical → most_frequent imputer + OneHotEncoder
- Models:
  - Baseline: Logistic Regression
  - Stronger: RandomForestClassifier + RandomizedSearchCV
  - SMOTE applied inside pipeline for class imbalance
- Threshold chosen based on **F1 score**.

## 5️⃣ Metrics (RandomForest)
| Metric | Score |
|--------|-------|
| Accuracy | 0.85 |
| Precision | 0.74 |
| Recall | 0.65 |
| F1-score | 0.69 |
| ROC-AUC | 0.83 |

*(Replace with actual scores after running your notebook)*

## 6️⃣ Feature Importance & Interpretability
- Top 5 drivers of churn:
  1. Contract_Term (Month-to-month)
  2. tenure
  3. MonthlyCharges
  4. PaperlessBilling
  5. InternetService_Fiber optic
- SHAP plots used to explain local and global feature impact.

## 7️⃣ Streamlit Demo
Run interactive churn predictor:

```bash
streamlit run streamlit_app.py
