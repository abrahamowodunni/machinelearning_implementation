import pickle
from flask  import Flask, request,app, jsonify, url_for, render_template

import pandas as pandas
import numpy as np 

app = Flask(__name__, template_folder= 'templates')

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

if __name__ =='__main__':
    app.run(debug=True)

