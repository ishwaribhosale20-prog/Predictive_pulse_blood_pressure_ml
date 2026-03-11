from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    systolic = float(request.form["systolic"])
    diastolic = float(request.form["diastolic"])
    bmi = float(request.form["bmi"])
    heart_rate = float(request.form["heart_rate"])

    features = np.array([[age, systolic, diastolic, bmi, heart_rate]])
    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "Normal"
    elif prediction[0] == 1:
        result = "Stage 1"
    else:
        result = "Stage 2"

    return render_template("index.html", prediction_text=f"Prediction : {result}")

if __name__ == "__main__":
    app.run(debug=True)
