import math
import numpy as np
from scipy.special import softmax

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
  if (activationFunction != 4):
    return (-1) * (target_j - output_j)
  else : # Help ini
    return "APA INIIIIII"

def derivate_Oj_To_NETj(output_j, activationFunction):
  if (activationFunction == 2):
    return output_j * (1 - output_j)
  elif (activationFunction == 1):
    return 1
  elif (activationFunction == 3):
    return 1 if (output_j) >= 0 else 0
  else : # Softmax buat target class, ryan tolong isi
    return 0 # Isi sama turunan softmax

def getErrorNodeOutput(target_j, output_j, activationFunction):
  return (-1) * (derivate_Ed_To_Oj(target_j, output_j, activationFunction) * derivate_Oj_To_NETj(output_j, activationFunction))

# For Hidden Layer
# d(Ed)/d(Wji) = d(Ed)/d(NETj) * d(NETj)/d(Wji)
# With d(Ed)/d(NETj) = d(Ed)/d(NETk) * d(NETk)/d(Oj) * d(Oj)/d(NETj)
# Thus -> d(Ed)/d(Wji) = d(Ed)/d(NETk) * d(NETk)/d(Oj) * d(Oj)/d(NETj)* d(NETj)/d(Wji)

# For Softmax
def derivate_Ed_To_NETj(activationFunction, sigma,  output_j): # Ini masih gatau bray
  return sigma * derivate_Oj_To_NETj(output_j, activationFunction)

def derivate_Ed_To_NETj_Softmax(pj, target_j):
  if (pj == target_j):
    return pj
  else:
    return -(1 - pj)

# def derivate_Ed_To_NETk(errorNode_k):
#   return (-1) * (errorNode_k)

# def derivate_NETk_To_Oj(w_kj):
#   return w_kj

# BACA INI
# Get Error Node for Hidden Neuron bikin di MBGD karena ada softmax