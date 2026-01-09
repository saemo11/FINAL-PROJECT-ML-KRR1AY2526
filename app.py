from flask import Flask, render_template, request
import pickle
import pandas as pd  # <-- added for DataFrame
import os

app = Flask(__name__)

# ---------------------------
# Load models
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr_model_path = os.path.join(BASE_DIR, "model", "lr_model.pkl")
rf_model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")

with open(lr_model_path, "rb") as f:
    lr_model = pickle.load(f)

with open(rf_model_path, "rb") as f:
    rf_model = pickle.load(f)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        humidity = float(request.form["humidity"])
        wind_speed = float(request.form["wind_speed"])
        rainfall = float(request.form["rainfall"])
        unit = request.form.get("unit", "C")  # default Celsius

        # Make sure column names match your model
        features_df = pd.DataFrame([[humidity, wind_speed, rainfall]],
                                   columns=["Humidity", "WindSpeed", "Rainfall"])

        # Predict
        lr_pred = lr_model.predict(features_df)[0]
        rf_pred = rf_model.predict(features_df)[0]

        prediction = (lr_pred + rf_pred) / 2

        if unit.upper() == "F":
            prediction = (prediction * 9/5) + 32

        final_prediction = round(prediction, 2)

        return render_template("index.html", prediction=final_prediction, unit=unit.upper())

    except Exception as e:
        return render_template("index.html", error=str(e), unit=request.form.get("unit", "C"))

if __name__ == "__main__":
    print("STARTING FLASK SERVER...")
    print("http://127.0.0.1:5000/")
    app.run(debug=True)
