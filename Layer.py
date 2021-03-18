from Neuron import Neuron
from utils import linear, sigmoid, reLu, softMax
class Layer:
  
  def __init__(self, bias):
    self.neuron_array = []
    self.input_array = []
    self.output_array = []
    self.type = 1 # default Linear
    self.bias = bias
    self.activationFunction = linear

  def setType(self, layerType):
    self.type = layerType
    if(layerType == 1):
      self.activationFunction = linear
    elif(layerType == 2):
      self.activationFunction = sigmoid
    elif(layerType == 3):
      self.activationFunction = reLu
    elif(layerType == 4):
      self.activationFunction = softMax
    else:
      self.activationFunction = sigmoid
    return self

  def getType(self):
    return self.type

  def createNeuron(self):
    newNeuron = Neuron(self.type, self.bias)
    self.neuron_array.append(newNeuron)
    return self
  
  def getNeuron(self, index):
    if(index < len(self.neuron_array)):
      return self.neuron_array[index]
    else:
      return None
  
  def getNeuronArray(self):
    return self.neuron_array
  
