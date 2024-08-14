#!/usr/bin/python3

import jetson_inference
import jetson_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="file name of the image to process")
parser.add_argument("--network", type=str, default="resnet-18", help="model to use, can be: googlenet, resnet-18, etc.")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print(f"Image recognized as: {class_desc} ({round(confidence *100, 4)}% #{class_idx})")