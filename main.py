import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Load dataset
df = pd.read_csv("insurance_claims.csv")

# Encode all object columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split X and Y
X = df.drop("fraud_reported", axis=1)
Y = df["fraud_reported"]
columns = X.columns
# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Model Accuracy:", accuracy)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, y_pred)
print("Model Accuracy:", accuracy)

import pickle

# Save model
pickle.dump(model, open("fraud_model.pkl","wb"))
# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and Scaler saved successfully!")

import joblib

# Save column names
columns = X.columns
joblib.dump(columns, "columns.pkl")

print("Columns saved successfully!")
