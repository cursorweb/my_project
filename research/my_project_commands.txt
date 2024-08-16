echo for reference only

exit

# python3 my_project.py
# --model=../jetson-inference/python/training/classification/models/cat_dog/resnet18.onnx
# --input_blob=input_0
# --output_blob=output_0
# --labels=../jetson-inference/python/training/classification/data/cat_dog/labels.txt
# ../jetson-inference/python/training/classification/data/cat_dog/test/cat/01.jpg
# cat.jpg

imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cat/01.jpg cat.jpg


# Things to run:
cd jetson-inference
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
# ./docker/run.sh
# use the hotfix version which contains updated code
./docker/run.sh --volume /home/nvidia/jetson-inference/python/training/classification:/jetson-inference/python/training/classification
cd python/training/classification/



# accuracy too low
python3 train.py --batch-size=2 --workers=1 --epochs=16 --model-dir=models/cars_type data/cars_type

# default command
python3 train.py --model-dir=models/cars_type data/cars_type

# export
python3 onnx_export.py --model-dir=models/cars_type


# retrain from
python3 train.py --model-dir=models/cars_type data/cars_type --resume models/cars_type/checkpoint.pth.tar
python3 train.py --model-dir=models/cars_type data/cars_type --resume models/cars_type/model_best.pth.tar --start-epoch 0 --epochs 70

# export
python3 onnx_export.py --model-dir=models/cars_type


# copy
cp jetson-inference/python/training/classification/models/cars_type/resnet18.onnx 