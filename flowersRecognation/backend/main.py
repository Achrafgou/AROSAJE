from fastapi import FastAPI
import tensorflow as tf
from tensorflow.python.framework import ops
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import os
from pydantic import BaseModel
import cv2
import base64

from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt


origins = ["*"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


labels = os.listdir("../data")




class Item(BaseModel):
    img: str

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_cropped(img):
    faces = face_classifier.detectMultiScale(img, 1.1, 9)
    if faces is ():
        return None
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]
    return cropped_face


@app.post("/")
async def root(item :Item):
    ops.reset_default_graph()

    # tf.reset_default_graph()
    convnet = input_data(shape=[50,50,1])

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    # 32 filters and stride=5 so that the filter will move 5 pixel or unit at a time
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)



    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate = 0.001, loss='categorical_crossentropy')

    model = tflearn.DNN(convnet, tensorboard_verbose=1)
    model.load("my_model.tflearn")

    nparr = np.fromstring(base64.b64decode(item.img), np.uint8)
    img_data = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(face_cropped(img_data), (50,50))

    pred = model.predict([np.array(img_data).reshape(50,50,1)])
    print(pred)
    print(np.argmax(pred))
    print(labels[np.argmax(pred)])

    return {"message": "Hello World"}
