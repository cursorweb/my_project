# 18+ file
# from os import listdir
# from os.path import isfile, join
import os
import sys

# print('truck', 'train', 'car', 'motorcycle', 'bus')

name = "val/bus"
path = os.path.abspath(f"../jetson-inference/python/training/classification/data/cars_type/{name}")

onlyfiles = os.listdir(path)

def get_size(p, f):
    return os.path.getsize(os.path.join(p, f))

empty_count=0
for file in onlyfiles:
    if get_size(path, file) == 0:
        empty_count+=1
print(empty_count, "empty files detected")
if input("continue?").lower() == "n":
    print("aborted")
    sys.exit(0)

for file in onlyfiles:
    # print(get_size(path, file))
    if get_size(path, file) == 0:
        os.remove(os.path.join(path, file))