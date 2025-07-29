from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("university_rank_model.h5")
scaler = joblib.load("university_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            tuition = float(request.form["tuition"])
            enrollment = float(request.form["enrollment"])
            
            input_data = np.array([[tuition, enrollment]])
            input_scaled = scaler.transform(input_data)
            
            rank = model.predict(input_scaled)[0][0]
            prediction = f"Predicted University Rank: {int(round(rank))}"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
