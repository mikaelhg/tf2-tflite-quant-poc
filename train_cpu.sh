#!/bin/bash

exec docker run --gpus all -it --rm \
   -w /app -v `pwd`:/app \
   -u 1000:1000 \
   tensorflow/tensorflow:2.0.0-py3 \
   python src/mnist.py
