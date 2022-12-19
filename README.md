# object-detection using TFLite

Goal: 
To build/train custom object detection model using TensorFlow Lite framework

Build dataset:
You can use LabelImg for tagging the objects in the image
Refer the link below for installation and dataset preparation
https://github.com/heartexlabs/labelImg

Steps to execute:
1. Set the path to current directory in command prompt
2. Command to run the complete process: docker compose up
3. Command to build the image from DockerFile: docker build <preferred container name> .
4. Command to run the container image in interactive type: docker run -it <assigned container name> bash
