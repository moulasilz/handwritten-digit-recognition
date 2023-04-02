import os
import numpy as np
from flask import Flask, flash, request, render_template, redirect, url_for, make_response
import pickle
from werkzeug.datastructures import  FileStorage
from werkzeug.utils import secure_filename
from PIL import Image as img
import cv2

#Create an app object using the Flask class. 
app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict(filename):
    image_arrays = np.array([[[0 for i in range(28)] for i in range(28)]])
    image = img.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print(image.size)
    if image.size != [28,28]:
        image = image.resize((28,28))
        print(image.size)
    gray = np.array(image)[:,:,0]
    gray_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    image_arrays[0] = np.array(gray)
    print(image_arrays[0])
    image_arrays = image_arrays.reshape((len(image_arrays), -1))
    print(len(image_arrays))
    predict = model.predict(image_arrays)
    return predict[0]

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded. Predict: ' + str(predict(filename)))
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/setcookie')
def setcookie(filename):
    resp = make_response(redirect("/"))
    resp.set_cookie('filename',filename)
    return resp

@app.route('/getcookie/<param>')
def getcookie(param):
    name = request.cookies.get(param)
    return name

@app.route('/draw')
def drawing():
    return render_template("draw.html")

from PIL import Image as img , ImageOps
import base64
import time

@app.route('/draw',methods=['POST'])
def upload_draw_image():
    if 'file' not in request.form:
        flash('No file part')
        return redirect(request.url)
    file = request.form['file']	
    image_filename = secure_filename(str(time.time()) + '.jpg')
    image_data = base64.b64decode(file)
    handler = open(os.path.join(app.config['UPLOAD_FOLDER'], image_filename), "wb+")
    handler.write(image_data)
    handler.close()
    
    image = img.open(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
    new_image = img.new("RGBA", image.size, "WHITE") # Create a white rgba background
    new_image.paste(image, (0, 0), image)              # Paste the image on the background. Go to the links given below for details.
    new_image = ImageOps.invert(new_image.convert('RGB'))
    new_image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename), "JPEG")
    prediction = predict(image_filename)
    flash('Image successfully uploaded. Predict: ' + str(prediction))
    return render_template("draw.html")
    # return redirect(location='/draw', Response= 'Image successfully uploaded. Predict: ' + str(prediction))


if __name__ == "__main__":
    app.run(debug = True)
    

