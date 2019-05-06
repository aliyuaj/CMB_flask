from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
app = Flask(__name__)
modelfile = 'predictor.pickle'
model = p.load(open(modelfile, 'rb'))
@app.route('/predict', methods=['POST'])
def make_prediction():
    message = request.get_json(force=True)
    prediction = model.predict(data)
    print(data)
    return jsonify(prediction[0])

if __name__ == '__main__': 
    app.run(debug=False, port='5000')