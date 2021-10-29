import numpy as np
from flask import Flask, request, jsonify, render_template
# import pickle
from keras.models import load_model
import cv2

app = Flask(__name__)
model = load_model('model3.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    file = request.form.items()
    img = cv2.imread(file)

    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(img))


if __name__ == "__main__":
    app.run(debug=True)
