import base64
import io
from flask import Flask,render_template
from flask import request
from flask import jsonify
import pickle as p
import numpy as np
from PIL import Image
import keras 
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing.image import img_to_array
from flask_cors import CORS, cross_origin
app = Flask(__name__)
#modelfile = 'model.pkl'
global model
model = load_model("CMB_model_TL.h5")
model._make_predict_function()
def img_preprocess(image,target_size):
	if image.mode !="RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image=img_to_array(image)
	image = np.expand_dims(image,axis=0)
	return image
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})
@app.route('/CMB')
def index():
    return render_template("index.html")
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])	
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict', methods=['POST','GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def predict():
	message = request.get_json(force=True)
	encoded_img = message['image']
	decoded_img =base64.b64decode(encoded_img)
	img=Image.open(io.BytesIO(decoded_img))
	processed_img= img_preprocess(img,target_size=(41,41))
	prediction = model.predict(processed_img).tolist()
	response = {
		'CMB': prediction[0]
	}
	return jsonify(response)
