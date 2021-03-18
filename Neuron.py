from utils import arrayMultiplication

class Neuron:
  
  def __init__(self, actType, bias=1):
    self.weight = []
    self.bias = bias

  def setWeight(self, weight):
    self.weight = weight

  def getWeight(self, index):
    return self.weight[index]
  
  def getWeightArray(self):
    return self.weight