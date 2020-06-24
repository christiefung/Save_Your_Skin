import tensorflow as tf
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image





def predict(img, d_model):
    model = tf.keras.models.load_model(d_model)

    img = img.convert("RGB")
    img = img.resize((64,64), Image.ANTIALIAS)


    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    result = prediction[0]
    return np.argmax(result)