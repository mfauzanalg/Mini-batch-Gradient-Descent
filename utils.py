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

# For Output Layer
# d(Ed)/d(Wji) = d(Ed)/d(Oj) * d(Oj)/d(NETj) * d(NETj)/d(Wji)
def derivate_Ed_To_Oj(target_j, output_j, activationFunction):
  if (activationFunction != softMax):
    return (-1) * (target_j - output_j)
  else :
    return softMax

def derivate_Oj_To_NETj(output_j, activationFunction):
  if (activationFunction == sigmoid):
    return output_j * (1 - output_j)
  elif (activationFunction == linear):
    return 1
  elif (activationFunction == reLu):
    return 1 if (output_j) >= 0 else 0
  else : # Softmax
    return 0

def derivate_NETj_To_Wji(x_ji):
  return x_ji

# For Hidden Layer
# d(Ed)/d(Wji) = d(Ed)/d(NETj) * d(NETj)/d(Wji)
# With d(Ed)/d(NETj) = d(Ed)/d(NETk) * d(NETk)/d(Oj) * d(Oj)/d(NETj)
# Thus -> d(Ed)/d(Wji) = d(Ed)/d(NETk) * d(NETk)/d(Oj) * d(Oj)/d(NETj)* d(NETj)/d(Wji)

# For Softmax
def derivate_Ed_To_NETj(activationFunction, pj = None, errorNode_k = None, w_kj = None, output_j = None, x_ji = None): # Ini masih gatau bray
  if (activationFunction == softMax):
    return pj
  else :
    return derivate_Ed_To_NETk(errorNode_k) * derivate_NETk_To_Oj(w_kj) * derivate_Oj_To_NETj(output_j, activationFunction)


def derivate_Ed_To_NETk(errorNode_k):
  return (-1) * (errorNode_k)

def derivate_NETk_To_Oj(w_kj):
  return w_kj

# def derivate_Oj_To_NETj(output_j, activationFunction):
# Same derivate with the output layer

def derivate_NETj_To_Wji(x_ji):
  return x_ji