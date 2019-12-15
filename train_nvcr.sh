#!/bin/bash

exec docker run --gpus all -it --rm \
   -w /app -v `pwd`:/app \
   -u 1000:1000 \
   --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
   nvcr.io/nvidia/tensorflow:19.11-tf2-py3 \
   python src/mnist.py
