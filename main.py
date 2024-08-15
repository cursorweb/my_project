#!/usr/bin/python3
import jetson_inference
import jetson_utils
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="file name of the image to process")
parser.add_argument("--save", type=str, default='none', nargs='?', help="Save a resulting image")
opt = parser.parse_args()

NETWORK = os.path.join("test_model.onnx")
LABELS = os.path.join("model/labels.txt")

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(
    NETWORK,
    labels=LABELS,
    input_blob="input_0",
    output_blob="output_0"
)

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
confidence = round(confidence * 100, 2)

# 'none' is the default if no argument was passed
# None is if the user didn't specify a file name, but the flag exists
if opt.save != 'none':
    font = jetson_utils.cudaFont()
    font.OverlayText(img, text=f"{confidence}% {class_desc}", 
                     x=5, y=5,
                     color=font.White, background=font.Gray40)
    if opt.save == None:
        file_name = f"{os.path.splitext(os.path.basename(opt.filename))[0]}.class.jpg"
    else:
        file_name = opt.save

    output = jetson_utils.videoOutput(file_name)

    output.Render(img)


print(f"\n\n\n{'-' * 10} Results {'-' * 10}")
print(f"Image recognized as: {class_desc} ({confidence}%)")