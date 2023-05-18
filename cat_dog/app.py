


from unittest import result
from flask import Flask,render_template,request,redirect,flash,url_for,send_file

from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf

from tensorflow import keras
model = keras.models.load_model('./model_animal_cnn_1.h5')
import cv2
import os
test_data = []
app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')

def index():
    return render_template('index.html')
@app.route('/',methods=['GET','POST'])

def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename= secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            file1 =os.path.join(app.config['UPLOAD_FOLDER'],filename)
            test_img_o = cv2.imread(file1)
            HEIGHT =32
            WIDTH = 55
            N_CHANNELS = 3
            test_data = []
            test_img = cv2.resize(test_img_o,(WIDTH,HEIGHT))
            test_data.append(test_img)
            test_data = np.array(test_img,dtype="float")/255.0
            test_data = test_data.reshape([-1,32,55,3])
            pred = model.predict(test_data)
            from numpy import argmax
            predictions = argmax(pred,axis=1)
            categories = ['dog','panda','cat']
            for idx,animal, x in zip(range(0,3),categories, pred[0]):
                print("ID: {}, Label: {} -> {}%".format(idx,animal,round(x*100,2)))
                print('Prediction : '+categories[predictions[0]])
            result = []
            result.append('Prediction :'+categories[predictions[0]])
            result.append(file1)
            
            #return send_file(filename, mimetype='image/gif')
            return render_template('result.html',value = result)
   # return render_template('index.html')
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
if __name__ == '__main__':
    
    app.run(debug= True)