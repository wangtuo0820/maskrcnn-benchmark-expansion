#!/bin/bash
ln -s ../maskrcnn_benchmark/engine/*.py ./
rm __init__.py
ln -s ../maskrcnn_benchmark/config/*.py ./
rm __init__.py
