import pickle
from flask  import Flask, request,app, jsonify, url_for, render_template

import pandas as pandas
import numpy as np 

app = Flask(__name__, template_folder= 'templates',static_url_path='/static')

regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data'] # we are trying to get the predict data in a json form
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    rounded_output = round(output[0], 2)
    print(rounded_output)
    return jsonify(rounded_output)

@app.route('/predict', methods=['POST'])

def predict():
    data=[float(x) for x in request.form.values()] # so what ever we input in our form becomes float 
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)
    rounded_output = round(output[0], 2) * 1000
    return render_template('home.html',prediction_text = f'The house price prediction is ${rounded_output}')

if __name__ =='__main__':
    app.run(debug=True)

