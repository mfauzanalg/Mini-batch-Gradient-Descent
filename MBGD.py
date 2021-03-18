from Layer import Layer
from utils import arrayMultiplication, derivate_Ed_To_Oj, derivate_Oj_To_NETj, derivate_NETj_To_Wji, derivate_Ed_To_NETj, derivate_Ed_To_NETk, derivate_NETk_To_Oj, derivate_NETj_To_Wji
import json
class MBGD:
  
  # Masih kurang struktur jaringannya (jumlah layer, jumlah neuron setiap layer, fungsi aktivasi setiap layer)
  def __init__(self, learning_rate=0.1, error_threshold=1, max_iter=200, batch_size=0, initial_weight=0.5):
    self.learning_rate = learning_rate
    self.error_threshold = error_threshold
    self.max_iter = max_iter
    self.batch_size = batch_size
    self.initial_weight = initial_weight
    self.epoch = None
    self.error = None
    self.bias = None
    self.layerArray = None
    
  def setBias(self, bias):
    self.bias = bias
    return self
  
  def getBias(self):
    return self.bias
  
  def getLayer(self, index):
    if(len(self.layerArray) > index):
      return self.layerArray[index]
    else:
      return None

  def createHiddenLayer(self, nNeuron, activation):
    newLayer = Layer(self.bias)
    newLayer.setType(activation)
    for i in range (nNeuron):
      newLayer.createNeuron()
    return newLayer

  def createOutputLayer(self, activation):
    newLayer = Layer(self.bias)
    newLayer.setType(activation)
    newLayer.createNeuron()
    return newLayer

  def setLayer(self, layers):
    newLayer = Layer(self.bias)
    newLayer.setType(0)
    self.layerArray = layers
    self.layerArray.insert(0,newLayer)

  def fit(self, dataset, label):
    while (self.epoch < self.max_iter or self.error > self.error_threshold):
      n_processed = 0
      update_weight = False

      for index_data, data in enumerate(dataset):
        # FORWARD PROPAGATION
        # result_output_layer = None
        prevLayerInputArray = data

        #if n process%batchsize = 0 updet w
        if (n_processed % self.batch_size == 0):
          for layer in self.layerArray:
            for neuron in layer.getNeuronArray():
              accumulative_delta = neuron.getAccumulativeDelta()
              for index_weight, weight in enumerate(neuron.getWeightArray()):
                weight += accumulative_delta[index_weight]

        for layerNum in range(len(self.layerArray)):
          layer = self.layerArray[layerNum]
          nextLayerInputArray = []
          nextLayerInputArray.append(self.bias)

          # Input layer
          if (layer.getType() == 0):
            nextLayerInputArray.extend(data)
            prevLayerInputArray = nextLayerInputArray.copy()
            
          # hidden layer
          else :
            for neuron_idx in range(len(layer.getNeuronArray())):
              neuron = layer.getNeuron(neuron_idx)

              if (neuron.getWeightArray() == None):
                neuron.setWeight([self.initial_weight * len(prevLayerInputArray)])
              
              sigma = arrayMultiplication(prevLayerInputArray, neuron.getWeightArray(), len(prevLayerInputArray))
              nextLayerInputArray.append(sigma)
              # update weight (lRate * (yi - oi)*xi)
              
              # Last Neuron
              if (neuron_idx == (len(layer.getNeuronArray())-1)):
                # activate and append to nextLayer
                nextLayerInputArray.pop(0)
                nextLayerInputArray = layer.activationFunction(nextLayerInputArray) # oi arr
                nextLayerInputArray.insert(0, self.bias)
                prevLayerInputArray = nextLayerInputArray

        # BACKWARD
        targetMinOutputArr = []
        for index_output in range(len(nextLayerInputArray)):
          targetMinOutputArr.append(label[index_data] - nextLayerInputArray[index_output])
        targetMinOutputArr.pop(0)

        ErrorNodeOutput = 3 # gatau fungsinya apa
        
        
        # Last Layer (Output Layer)
        # if (layerNum == (len(self.layerArray)-1)):
        #   result_output_layer = prevLayerInputArray
      



  # INI BEKAS YG FFNN
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

    