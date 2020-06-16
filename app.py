import os
from werkzeug.utils import secure_filename
import uuid
import base64
import requests
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import imutils
import numpy as np

model = tf.keras.models.load_model('/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/models/model.h5')

UPLOAD_FOLDER = '/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

# @app.route('/')
# def index():
#    return render_template('index.html')

def predict(file):
    x = load_img(file, target_size=(224, 224))

    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
#    images = np.vstack([x])
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Label: Dry")
    elif answer == 1:
        print("Label: Normal")
    elif answer == 2:
        print("Label: Oily")
    return answer


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
    return render_template('template.html', label='',
                           imagesource='/Users/ChristieFung/Desktop/Insight/skin_care_scraper-2/uploads/template.jpg')


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
            result = predict(file_path)
            if result == 0:
                label = 'Dry'
            elif result == 1:
                label = 'Normal'
            elif result == 2:
                label = 'Oily'
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str(time.time() - start_time))
            # return render_template('template.html', label=label, imagesource='../uploads/' + filename)
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)


from flask import send_from_directory


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.debug = True
    app.run()
