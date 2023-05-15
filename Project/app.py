from __future__ import division, print_function
import sys, glob, re
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import numpy as np
import os
import uuid
import urllib
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'model.hdf5'))
model.make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size = (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)

    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('start.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)

        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        #result = pred_class[0][0][1]

        return result
        

    return None

if __name__ == '__main__':
    app.run(debug=True)