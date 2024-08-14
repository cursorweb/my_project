import os
import sys

# print('truck', 'train', 'car', 'motorcycle', 'bus')

name = "train/train"
path = os.path.abspath(f"../jetson-inference/python/training/classification/data/cars_type/{name}")

onlyfiles = os.listdir(path)

def get_size(p, f):
    return os.path.getsize(os.path.join(p, f))

empty_count=0
for file in onlyfiles:
    empty_count += 1
    # if get_size(path, file) == 0:
    #     empty_count += 1
print(empty_count, "files detected")
sys.exit(0)
# if input("continue?").lower() == "n":
#     print("aborted")
#     sys.exit(0)

for file in onlyfiles:
    # print(get_size(path, file))
    if get_size(path, file) == 0:
        os.remove(os.path.join(path, file))