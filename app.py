import os
import requests
from PIL import Image
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import uuid
import base64

fn = 'lr_skin_classification'
model_dir = '/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/models'
with open('%s/%s.pkl' % (model_dir, fn), 'rb') as f:
    model = pickle.load(f)

UPLOAD_FOLDER = '/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)


# @app.route('/')
# def index():
#    return render_template('index.html')

def predict(file):
    image_size = (64, 64)
    image = np.asarray(Image.open(file).resize(image_size))
    image = np.average(image, axis=2)
    image /= 255
    dimension = image_size[0] * image_size[1]
    image = image.reshape(1, dimension)
    array = model.predict(image)
    result = array[0]
#    prob = model.predict_proba(image)
#    prob = model.predict(image)

    if result == 0:
        print("Label: Normal")
    elif result == 1:
        print("Label: Dry")
    elif result == 2:
        print("Label: Oily")
    return result
#    return prob

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]  # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('template.html', label=' ',
                           imagesource='/Users/ChristieFung/Documents/Save_Your_Skin/uploads/template.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_size = (64, 64)
            image = np.asarray(Image.open(file).resize(image_size))
            image = np.average(image, axis=2)
            image /= 255
            dimension = image_size[0] * image_size[1]
            image = image.reshape(1, dimension)
            array = model.predict(image)
            result = array[0]
#            prob = model.predict_proba(image)
 #           prob = model.predict(image)
#            result = model.predict(file_path)

            if result == 0:
                label = 'Normal'
            elif result == 1:
                label ='Dry'
            elif result == 2:
                label = 'Oily'

            print(result)
#            print(prob)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str(time.time() - start_time))
#            return render_template('template.html', label = label, prob = prob, imagesource='../uploads/' + filename)
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.debug = True
    app.run()
