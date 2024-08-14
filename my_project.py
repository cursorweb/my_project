#!/usr/bin/env python3
# python3 my_project.py --model=../jetson-inference/python/training/classification/models/cat_dog/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=../jetson-inference/python/training/classification/data/cat_dog/labels.txt ../jetson-inference/python/training/classification/data/cat_dog/test/cat/01.jpg cat.jpg
# imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cat/01.jpg cat.jpg


# accuracy too low
# python3 train.py --batch-size=2 --workers=1 --epochs=4 --model-dir=models/cars_type data/cars_type

# default command
# python3 train.py --model-dir=models/cars_type data/cars_type

# export
# python3 onnx_export.py --model-dir=models/cars_type


# retrain from
# python3 train.py --model-dir=models/cars_type data/cars_type --resume 

# export retrain from
# python3 onnx_export.py --model-dir=models/cars_type --input checkpoint.pth.tar

# export
# python3 onnx_export.py --model-dir=models/cars_type


import jetson_inference
import jetson_utils

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("filename", type=str, help="file name of the image to process")
# parser.add_argument("--network", type=str, default="resnet-18", help="model to use, can be: googlenet, resnet-18, etc.")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print(f"Image recognized as: {class_desc} ({round(confidence *100, 4)}% #{class_idx})")