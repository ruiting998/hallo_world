

import itertools
import os

import matplotlib.pylab as plt
import numpy as np

#import tensorflow as tf

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import cv2

is_init = False

global module_selection
global handle_base
global IMAGE_SIZE
global datagen_kwargs
global dataflow_kwargs
global valid_datagen


data_dir = 'train/'
BATCH_SIZE = 32
def init_img_classify():

    global new_model
    new_model = tf.keras.models.load_model('saved_arm_model')

    module_selection = ("mobilenet_v2_100_224", 224)
    handle_base, pixels = module_selection

    IMAGE_SIZE = (pixels, pixels)

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")


    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    global valid_generator
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=False, **dataflow_kwargs)


def get_class_string_from_index(index):
    for class_string, class_index in valid_generator.class_indices.items():
        if class_index == index:
            return class_string

def get_image_classifal(image):
    from datetime import datetime
    ts = datetime.now()
    global predicted_class
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image,(224,224))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image = (image/255.0)*0.2

    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = new_model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    predicted_class = predicted_index

    te = datetime.now()
    print(f'time cost: {(te-ts).microseconds/1000} ms')
    print("Predicted label: " + get_class_string_from_index(predicted_index))

    return predicted_class





