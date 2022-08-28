'''
from tensorflow.python.client import device_lib
import torch
import keras
print(device_lib.list_local_devices())
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(keras.__version__)'''
import dlib
import inspect
print(inspect.getfile(dlib))