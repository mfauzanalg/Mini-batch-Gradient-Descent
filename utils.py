import math
import numpy as np

def linear(arr):
  res = []
  for el in arr :
    res.append(el)
  return res

def sigmoid(arr):
  res = []
  for el in arr:
    res.append(1 / (1 + math.exp(-el)))
  return res

def reLu(arr):
  res = []
  for el in arr:
    res.append(max(0, el))
  return res
  
def softMax(x):
  e_x = np.exp(x - np.max(x))
  return (e_x / e_x.sum()).tolist()

def arrayMultiplication(x_array, w_array, length):
  oi = 0
  for i in range(length):
    oi += x_array[i] * w_array[i]
  return oi