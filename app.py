import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D, Concatenate, Input, Add, Dense, Activation, Flatten, Conv2D, Conv2DTranspose, Dropout
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import flask
from flask import Flask, jsonify, request
from flask import render_template


app = Flask(__name__)


def load_image(image):
    '''this function will resize the images to 1024*1024
       and normalize every pixel by dividing by 255'''
    image = cv2.imread(image)
    image = tf.image.resize(image, [1024,1024])
    image = tf.cast(image, tf.float32) / 255.0
    return image 


interpreter = tf.lite.Interpreter("vgg_compressed.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def create_mask(col_pred_mask, table_pred_mask):
  col_pred_mask = tf.argmax(col_pred_mask, axis = -1)
  col_pred_mask = col_pred_mask[..., np.newaxis]
  table_pred_mask = tf.argmax(table_pred_mask, axis = -1)
  table_pred_mask = table_pred_mask[..., np.newaxis]
  return col_pred_mask[0], table_pred_mask[0]


pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def image_to_text(img_name):
    text = pytesseract.image_to_string(Image.open(img_name))
    return text


def make_prediction(image):
    
    ## resize and preprocess the image
    image = load_image(image)
    image = image[np.newaxis, :,:,:]
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    col_mask = interpreter.get_tensor(output_details[0]['index'])
    tab_mask = interpreter.get_tensor(output_details[1]['index'])
    col_mask, tab_mask = create_mask(col_mask, tab_mask)
    image = tf.squeeze(image)
    im = tf.keras.preprocessing.image.array_to_img(image)
    im.save("static/original_image.bmp")
    im = tf.keras.preprocessing.image.array_to_img(tab_mask)
    im.save("static/tab_mask.bmp")
    img = Image.open("static/original_image.bmp")
    table_mask = Image.open("static/tab_mask.bmp")
    img.putalpha(table_mask)
    img.save("static/output_alpha.png")
    text = (image_to_text("static/output_alpha.png"))
    return text

@app.route('/index')
def index():
    return flask.render_template('deploy.html')

@app.route('/predict', methods = ['POST'])
def predict():
    image = request.files['filename']
    image.save("test_image.jpeg")
    image_name = "test_image.jpeg"
    text = make_prediction(image_name)
    return flask.render_template("result.html", text = text )

if __name__ == "__main__":
    app.debug = True
    app.run(host = '0.0.0.0', port = 8080)








    