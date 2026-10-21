
# Healthcare Predictive Analytics
# Project: Diabetes Risk Prediction

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

# 1. Create output folder
os.makedirs("outputs", exist_ok=True)

# 2. Load dataset
df = pd.read_csv("data/diabetes.csv", header=None)

df.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nOutcome distribution:")
print(df["Outcome"].value_counts())

# 3. Remove duplicate records
df = df.drop_duplicates().copy()

# 4. Replace invalid zero measurements with missing values.
# Zero is valid for Pregnancies and Outcome, so retain those.
invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in invalid_zero_columns:
    df[col] = df[col].replace(0, np.nan)

print("\nMissing values after checking zero measurements:")
print(df.isnull().sum())

# 5. Separate predictors and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

# 6. Split into training and testing sets
# Stratification preserves the approximate outcome proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Build a machine learning pipeline
# Median imputation and scaling are fitted on training data only.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 8. Train the model
model.fit(X_train, y_train)

# 9. Generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 10. Evaluate model performance
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

print("\nMODEL EVALUATION")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Save evaluation metrics
pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Score"]
).to_csv("outputs/evaluation_metrics.csv", index=False)

# Save actual outcomes and predictions
results = pd.DataFrame({
    "Actual_Outcome": y_test,
    "Predicted_Outcome": y_pred,
    "Predicted_Probability": y_prob
})

results.to_csv("outputs/predictions.csv", index=False)

# 11. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)
plt.title("Diabetes Prediction - Confusion Matrix")
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# 12. ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Diabetes Prediction")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=300)
plt.show()

# 13. Feature coefficient visualization
# Coefficients show the direction of association in this model.
classifier = model.named_steps["classifier"]

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": classifier.coef_[0]
}).sort_values("Coefficient")

coefficients.to_csv(
    "outputs/feature_coefficients.csv",
    index=False
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=coefficients,
    x="Coefficient",
    y="Feature"
)
plt.title("Logistic Regression Feature Coefficients")
plt.tight_layout()
plt.savefig("outputs/feature_coefficients.png", dpi=300)
plt.show()

# 14. Save trained model
joblib.dump(model, "outputs/diabetes_model.joblib")

print("\nProject completed successfully!")
print("Check the outputs folder for results and charts.")
