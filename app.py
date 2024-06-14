from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
# import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model_vinal_1.jlb")

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/prediksi', methods=["POST"])
def prediksi():

    # data1 = float(request.form['pH'])
    # data2 = float(request.form['Temprature'])
    data3 = float(request.form['Fat'])
    data4 = float(request.form['Odor']) 
    data5 = float(request.form['Turbidity'])
    data6 = float(request.form['Taste'])
    data7 = float(request.form['Colour'])
    
   
    arr = np.array([[data3, data4, data5, data6, data7]])
    pred = model.predict(arr)

    susu = pred

    if pred == 2:
        susu = "Low";
    elif pred == 1:
        susu = "Medium";
    else:
        susu  = "High";
        
    return render_template('index.html', prediction = "{}".format(susu))
    

if __name__ == "main":
    app.run(debug=True)