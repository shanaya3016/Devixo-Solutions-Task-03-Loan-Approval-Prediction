import pandas as pd
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

# --- Data load aur cleaning ---
df = pd.read_csv("data/loan_approval_dataset.csv")
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df = df.drop_duplicates().drop(columns=["loan_id"])
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

# --- Feature engineering ---
asset_cols = ["residential_assets_value", "commercial_assets_value",
              "luxury_assets_value", "bank_asset_value"]
df["total_assets"] = df[asset_cols].sum(axis=1)
df["loan_to_income"] = df["loan_amount"] / df["income_annum"]
df["assets_to_loan"] = df["total_assets"] / df["loan_amount"]
df["income_per_dependent"] = df["income_annum"] / (df["no_of_dependents"] + 1)
df["monthly_payment"] = df["loan_amount"] / (df["loan_term"] * 12)

# --- Split, imputer, feature selection ---
X = df.drop(columns=["loan_status"])
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

selector = SelectKBest(f_classif, k=10).fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()].tolist()
X_train = X_train[selected_features]
X_test = X_test[selected_features]
print("Selected features:", selected_features)

scaler = StandardScaler().fit(X_train)

# --- Best model (tuned params) ---
model = GradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Test F1:", round(f1_score(y_test, pred), 4))
print("Test Accuracy:", round(accuracy_score(y_test, pred), 4))

# --- Save ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.joblib")
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(model, f)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(selected_features, "models/selected_features.joblib")
print("Models saved in models/ folder")