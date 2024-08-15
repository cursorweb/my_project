import os
import sys

# print('truck', 'train', 'car', 'motorcycle', 'bus')

# done: bus, car

name = "train/bus"
path = os.path.abspath(f"../../jetson-inference/python/training/classification/data/cars_type/{name}")

onlyfiles = os.listdir(path)

def get_size(p, f):
    return os.path.getsize(os.path.join(p, f))

file_count = 0
empty_count = 0
for file in onlyfiles:
    file_count += 1
    if get_size(path, file) == 0:
        empty_count += 1
print(file_count, "files detected for", name)
print(empty_count, "empty files")
sys.exit(0)
# if input("continue?").lower() == "n":
#     print("aborted")
#     sys.exit(0)

for file in onlyfiles:
    # print(get_size(path, file))
    if get_size(path, file) == 0:
        os.remove(os.path.join(path, file))