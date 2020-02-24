import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import cv2
import torch
import numpy as np
from net import Net
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            text = prediction(path)
            return "<h1>"+text+"</h1>"
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

def prediction(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    IMG_SIZE = 50
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    model = Net().to("cpu")
    model.load_state_dict(torch.load("Model"))
    model.eval()
    img_array = np.array(img)
    X = torch.Tensor(img_array).view(-1,50,50)
    net_out = model(X.view(-1,1,50,50))
    predicted_class = torch.argmax(net_out)
    return "Dog" if predicted_class else "Cat"
    
if __name__ == "__main__":
    app.run()