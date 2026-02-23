from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("best_rf_model.pkl")
imputer = joblib.load("imputer.pkl") 
scaler = joblib.load("scaler.pkl")

FEATURES =["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

def class_name(c):
    return ["Low", "Medium", "High"][c]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            x = np.array(values).reshape(1, -1)
            x_imp = imputer.transform(x)
            x_scaled = scaler.transform(x_imp)
            pred_class = int(model.predict(x_scaled)[0])
            prediction = class_name(pred_class)
        except Exception as e:
            error = str(e)
    return render_template("index.html",prediction=prediction,error=error,features=FEATURES)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
