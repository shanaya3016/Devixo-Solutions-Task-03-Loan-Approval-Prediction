## Part 5: Project Documentation

### 1. Dataset
- **Name:** Loan Approval Prediction Dataset (Kaggle, by Archit Sharma)
- **Size:** 4,269 rows, 13 columns (12 after dropping `loan_id`)
- **Target:** `loan_status` (Approved = 1, Rejected = 0), 62.2% Approved / 37.8% Rejected
- **Features:** dependents, education, self-employment, annual income, loan amount, loan term, CIBIL score, and four asset values
- **Quality:** 0 missing values, 0 duplicates

### 2. Workflow
1. Imported data, stripped extra spaces from column names, checked missing values and duplicates
2. Encoded categorical columns (education, self_employed)
3. **Feature Engineering:** created total_assets, loan_to_income, assets_to_loan, income_per_dependent, monthly_payment
4. 80/20 stratified train/test split, median imputer (reusable for missing values)
5. **Feature Selection:** SelectKBest (f_classif, k=10)
6. **Scaling:** StandardScaler (fitted on train only, to avoid leakage)
7. Trained 5 models, 5-fold cross validation, GridSearchCV tuning
8. Feature importance analysis, best model saved with Joblib and Pickle
9. Streamlit app for predictions

### 3. Model Selection
Trained Random Forest, Decision Tree, Gradient Boosting, XGBoost and SVM. Compared on Accuracy, Precision, Recall, F1 and ROC-AUC, then validated with 5-fold stratified cross validation and tuned with GridSearchCV (Random Forest, Gradient Boosting, Decision Tree, SVM).
Decision Tree and Gradient Boosting both reached 100% test F1. **Gradient Boosting** was chosen because an ensemble is more stable than a single tree, and its CV std was very low (0.0005).

### 4. Results
| Model | Best CV F1 | Test F1 | Test Accuracy |
|---|---|---|---|
| Gradient Boosting | 0.9998 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9988 | 0.9991 | 0.9988 |
| SVM | 0.9729 | 0.9804 | 0.9754 |

- Tuning improved SVM the most (F1 0.9693 to 0.9804).
- Feature importance (Gradient Boosting): cibil_score 81%, loan_to_income 9%, loan_term 8.5%, assets_to_loan 1.4%; the other features were close to 0.
- Correlation of cibil_score with loan_status was 0.77, the highest of all features.

### 5. Limitations
- Approval depends mostly on `cibil_score` (threshold about 550). A one-split tree gives 96.7% accuracy, so near-perfect scores reflect a simple dataset, not necessarily real-world performance.
- The dataset appears synthetic, so results may not transfer to real banking data.
- Several features are highly correlated (income, loan amount, assets), which is multicollinearity.
- SelectKBest with fixed k=10 kept some low-scoring features.
- Class imbalance is mild (62/38) and was not resampled.
- XGBoost was trained but not tuned (optional in the task).

### 6. Future Improvements
- Test on a larger, real-world loan dataset
- Tune XGBoost and try RandomizedSearchCV or Optuna
- Try SMOTE or class weights for imbalance
- Add model explainability (SHAP)
- Wrap everything in a single sklearn Pipeline for deployment
- Deploy the Streamlit app online (Streamlit Community Cloud)
## Screenshots

### Correlation Heatmap
![Heatmap](screenshots/heatmap.png)

### Model Comparison
![Model Comparison](screenshots/model_comparison.png)

### Feature Importance
![Feature Importance](screenshots/feature_importance.png)

### Streamlit App
![Approved](screenshots/streamlit_approved.png)
![Rejected](screenshots/streamlit_rejected.png)

---
*Final model was re-trained locally using `train_and_save.py` (same data, preprocessing and tuned parameters) to avoid a scikit-learn version mismatch between Colab and the local machine.*