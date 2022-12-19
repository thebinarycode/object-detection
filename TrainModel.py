# https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb

from contextlib import redirect_stdout
import os
import sys

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = tf.compat.v1.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.compat.v1.Session(config=config)

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    'dataset/train',
    'dataset/train',
    ['android', 'pig_android']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'dataset/validation',
    'dataset/validation',
    ['android', 'pig_android']
)


def train(model_name, batch_size, epochs, save_dir):
    # Model architecture	Size(MB)*	Latency(ms)**	Average Precision***
    # EfficientDet-Lite0	4.4	        146	            25.69%
    # EfficientDet-Lite1	5.8	        259	            30.55%
    # EfficientDet-Lite2	7.2	        396	            33.97%
    # EfficientDet-Lite3	11.4	    716	            37.70%
    # EfficientDet-Lite4	19.9	    1886	        41.96%

    spec = model_spec.get(model_name)

    model = object_detector.create(train_data, model_spec=spec, batch_size=batch_size, train_whole_model=True, epochs=epochs, validation_data=val_data)
    model.evaluate(val_data)

    with open(save_dir + '/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    model.export(export_dir=save_dir, tflite_filename='android.tflite')


    # Several factors can affect the model accuracy when exporting to TFLite:

    # Quantization helps shrinking the model size by 4 times at the expense of some accuracy drop.
    # The original TensorFlow model uses per-class non-max supression (NMS) for post-processing, while the TFLite model uses global NMS that's much faster but less accurate. Keras outputs maximum 100 detections while tflite outputs maximum 25 detections.
    # Therefore you'll have to evaluate the exported TFLite model and compare its accuracy with the original TensorFlow model.
    model.evaluate_tflite(save_dir + '/android.tflite', val_data)


if __name__ == "__main__":
    # efficientdet_lite0 4 10 saved_model
    print("Inside train model script")
    print(f"args received: {sys.argv[1:]}")
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    save_dir = sys.argv[4]

    #train("efficientdet_lite0", 1, 5, "saved_model")
    train(model_name, batch_size, epochs, save_dir)

