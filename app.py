from __future__ import division, print_function
import os
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'BreastCancerModels/model3.h5'

model = load_model(MODEL_PATH)
model.make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(100, 100))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    f = request.files['file']

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, secure_filename(f.filename))
    f.save(file_path)

    preds = model_predict(file_path, model)[0][0]
    preds = round(preds, 2)
    string = 'Some Error occurred'
    if preds > 0.5:
        string =  'Cancerous Tissue Detected'
    else:
        string =  'Non-Cancerous Tissue Detected'
    return string

if __name__ == '__main__':
    app.run(debug=True)

