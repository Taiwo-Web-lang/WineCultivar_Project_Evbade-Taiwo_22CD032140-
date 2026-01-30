from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "wine_cultivar_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["magnesium"]),
            float(request.form["color_intensity"]),
            float(request.form["proline"])
        ]

        scaled_features = scaler.transform([features])
        pred = model.predict(scaled_features)[0]

        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
