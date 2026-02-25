# ========================================
# INSURANCE FRAUD DETECTION - FLASK APP
# ========================================

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model, scaler, columns
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html", columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []

        for col in columns:
            value = request.form.get(col)
            input_data.append(float(value))

        input_array = np.array(input_data).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            result = "Fraudulent Claim ❌"
        else:
            result = "Genuine Claim ✅"

        return render_template("index.html",
                               columns=columns,
                               prediction_text=result)

    except Exception as e:
        return render_template("index.html",
                               columns=columns,
                               prediction_text="Error: Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
