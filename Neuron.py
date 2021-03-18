from utils import arrayMultiplication

class Neuron:
  
  def __init__(self, actType, bias=1):
    self.weight = []
    self.bias = bias
    self.accumulative_delta = []

  def setWeight(self, weight):
    self.weight = weight

  def setErrorNode(self, errorNode):
    self.errorNode = errorNode

  def getWeight(self, index):
    return self.weight[index]
  
  def getWeightArray(self):
    return self.weight
  
  def getAccumulativeDelta(self):
    return self.accumulative_delta
  
  def setAccumulativeDelta(self, accumulative_delta):
    self.accumulative_delta = accumulative_delta
  
  def getErrorNode(self):
    return self.errorNode
  