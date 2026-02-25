# ========================================
# INSURANCE FRAUD DETECTION - TRAIN MODEL
# ========================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("insurance_claims.csv")

# Drop unwanted columns
if 'policy_number' in df.columns:
    df.drop(columns=['policy_number'], inplace=True)

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Fill missing values
df.fillna(0)

# -------------------------
# ENCODE CATEGORICAL
# -------------------------
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# -------------------------
# SPLIT FEATURES & TARGET
# -------------------------
X = df.drop("fraud_reported", axis=1)
y = df["fraud_reported"]

# Save column names (important for Flask)
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#-------------------------
# CLEAN BEFORE SMOTE  👇 ADD THIS HERE
# -------------------------
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train=X_train.fillna(X_train.median(numeric_only=True))
for col in X_train.select_dtypes(include="object").columns:
   X_train[col]=X_train[col].fillna(X_train[col].mode()[0])

# -------------------------
# HANDLE IMBALANCE
# -------------------------
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# -------------------------
# SCALING
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# MODEL
# -------------------------
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# -------------------------
# SAVE FILES
# -------------------------
pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model & Scaler Saved Successfully!")
