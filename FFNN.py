from Layer import Layer
from utils import arrayMultiplication
import json
class FFNN:
  
  def __init__(self):
    self.layerArray = []
  
  def setBias(self, bias):
    self.bias = bias
    return self
  
  def getBias(self):
    return self.bias
  
  def createLayer(self):
    newLayer = Layer(self.bias)
    self.layerArray.append(newLayer)
    return self
  
  def getLayer(self, index):
    if(len(self.layerArray) > index):
      return self.layerArray[index]
    else:
      return None

  def loadModel(self, filename):
    file = open(filename, "r")
    json_file = file.read()
    layers = json.loads(json_file)
    # input layer
    for layer in layers:
      self.createLayer()
      newLayer = self.getLayer(len(self.layerArray)-1)
      newLayer.setType(layer['type'])
      for weight in layer['neurons']:
        newLayer.createNeuron()
        newNeuron = newLayer.getNeuron(len(newLayer.neuron_array)-1)
        newNeuron.setWeight(weight)
    return self

  def predict(self, dataset):
    output_sets = []
    for data in dataset:
      predicted_output = None

      prevLayerInputArray = data
      for layerNum in range(len(self.layerArray)):
        layer = self.layerArray[layerNum]
        nextLayerInputArray = []
        nextLayerInputArray.append(self.bias)

        if (layer.getType() == 0): #Input Layer
          nextLayerInputArray.extend(data)
          prevLayerInputArray = nextLayerInputArray.copy()
          
        else :
          for neuron_idx in range(len(layer.getNeuronArray())) :
            neuron = layer.getNeuron(neuron_idx)
            sigma = arrayMultiplication(prevLayerInputArray, neuron.getWeightArray(), len(prevLayerInputArray))
            
            # output = neuron.activationFunction(sigma)
            nextLayerInputArray.append(sigma)

            # Last Neuron
            if (neuron_idx == (len(layer.getNeuronArray())-1)):
              # activate and append to nextLayer
              nextLayerInputArray.pop(0)
              nextLayerInputArray = layer.activationFunction(nextLayerInputArray)
              nextLayerInputArray.insert(0, self.bias)
              prevLayerInputArray = nextLayerInputArray

        # Last Layer (Output Layer)
        if (layerNum == (len(self.layerArray)-1)):
          predicted_output = prevLayerInputArray

    # Input the Predicted output to output set
      output_sets.append(predicted_output[1])
      
    return output_sets

  def printmodel(self):
    i = 1
    tab = " " 
    for layer in self.layerArray:
      if (i == 1):
        print("Layer Input")
      elif (i == len(self.layerArray)):
        print("Layer output")
      else:
        print("Layer " + str(i))
      i += 1
      j = 0
      for neuron in layer.neuron_array:
        j += 1
        print (2*tab + "Neuron " + str(j))
        print(4*tab + "Weight: ", end="")
        print(neuron.weight)


  def classify(self, output, activation):
    # Sigmoid
    classified_output = []
    for i in range(len(output)):
      # Sigmoid
      if (activation == 2):
        if (output[i] >= 0.5):
          classified_output.append(1)
        else:
          classified_output.append(0)
      # Linear
      elif (activation == 1):
        if (output[i] >= 0.5):
          classified_output.append(1)
        else:
          classified_output.append(0)

    return classified_output

    