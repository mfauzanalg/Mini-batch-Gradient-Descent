from Layer import Layer
from utils import getErrorNodeOutput, arrayMultiplication, derivate_Ed_To_NETj
import json
class MBGD:
  
  # Masih kurang struktur jaringannya (jumlah layer, jumlah neuron setiap layer, fungsi aktivasi setiap layer)
  def __init__(self, learning_rate=0.1, error_threshold=1.0048, max_iter=2000, batch_size=0, initial_weight=0.5):
    self.learning_rate = learning_rate
    self.error_threshold = error_threshold
    self.max_iter = max_iter
    self.batch_size = batch_size
    self.initial_weight = initial_weight
    self.epoch = 0
    self.error = 0
    self.bias = 1
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
    n_processed = 0
    self.error = 999999999999999

    while (self.epoch < self.max_iter and self.error > self.error_threshold):
      # E = 0
      update_weight = False
      self.error = 0
      for index_data, data in enumerate(dataset):
        # FORWARD PROPAGATION
        # result_output_layer = None
        prevLayerInputArray = data

        #if n process%batchsize = 0 updet w
        if (self.batch_size != 0 and n_processed != 0 and n_processed % self.batch_size == 0):
          for layer in self.layerArray:
            for neuron in layer.getNeuronArray():
              accumulative_delta = neuron.getAccumulativeDelta()
              weight_arr = neuron.getWeightArray()
              for index_weight in range(len(weight_arr)):
                weight_arr[index_weight] += accumulative_delta[index_weight]
              neuron.accumulative_delta = []
                
        if(n_processed % len(dataset) == 0 and self.batch_size != len(dataset)):
          for layer in self.layerArray:
            for neuron in layer.getNeuronArray():
              accumulative_delta = neuron.getAccumulativeDelta()
              weight_arr = neuron.getWeightArray()
              for index_weight in range(len(weight_arr)):
                if (self.epoch == 2):
                  print(weight_arr[index_weight], "Before")
                  print(accumulative_delta)
                  print(accumulative_delta[index_weight], "ACCCC")
                weight_arr[index_weight] += accumulative_delta[index_weight]
                if (self.epoch == 2):
                  print(weight_arr[index_weight], "after")
                  print()
              neuron.accumulative_delta = []

        n_processed+=1

        for layerNum in range(len(self.layerArray)):
          layer = self.layerArray[layerNum]
          nextLayerInputArray = []
          nextLayerInputArray.append(self.bias)

          # Input layer
          if (layer.getType() == 0):
            nextLayerInputArray.extend(data)
            prevLayerInputArray = nextLayerInputArray.copy()
            layer.output_array =  nextLayerInputArray.copy()

          # hidden layer
          else :
            for neuron_idx in range(len(layer.getNeuronArray())):
              neuron = layer.getNeuron(neuron_idx)

              if (len(neuron.getWeightArray()) == 0):
                neuron.setWeight([self.initial_weight for i in range(len(prevLayerInputArray))])
              

              sigma = arrayMultiplication(prevLayerInputArray, neuron.getWeightArray(), len(prevLayerInputArray))
              nextLayerInputArray.append(sigma)
              # update weight (lRate * (yi - oi)*xi)
              
              # Last Neuron
              if (neuron_idx == (len(layer.getNeuronArray())-1)):
                # activate and append to nextLayer
                nextLayerInputArray.pop(0)
                
                nextLayerInputArray = layer.activationFunction(nextLayerInputArray) # oi arr
                nextLayerInputArray.insert(0, self.bias)
                self.layerArray[layerNum].output_array = nextLayerInputArray
                prevLayerInputArray = nextLayerInputArray

        # BACKWARD
        targetMinOutputArr = []
        for index_output in range(len(nextLayerInputArray)):
          # Target - Output
          targetMinOutputArr.append(label[index_data] - nextLayerInputArray[index_output])
        # Removing Bias
        targetMinOutputArr.pop(0)
        nextLayerInputArray.pop(0)

        for diff_idx in range(len(targetMinOutputArr)):
          diff = targetMinOutputArr[diff_idx]
          self.error += diff*diff/2

        for layerNum in range(len(self.layerArray) - 1, 0, -1):
          layer = self.layerArray[layerNum]

          # Pertama dia ada di output layer hitung error node output
          if (layerNum == len(self.layerArray) - 1):
            for neuron_idx in range(len(layer.getNeuronArray())):
              neuron = layer.getNeuron(neuron_idx)
              # Target, Output, Layer.type
              neuron.errorNode = getErrorNodeOutput(label[index_data], nextLayerInputArray[neuron_idx], layer.type)

          # Sekarang ada di layer hidden
          else:
            nextLayer = self.layerArray[layerNum + 1]
            for neuron_idx in range(0, len(layer.getNeuronArray())):
              neuron = layer.getNeuron(neuron_idx)
              
              if (layer.getActivationFunction() == 4): # Softmax
                neuron.errorNode = ""
                
              else:
                totalWkhDk = 0
                for nextLayerNeuron_idx in range(len(nextLayer.getNeuronArray())):
                  nextLayerNeuron = nextLayer.getNeuron(nextLayerNeuron_idx)
                  thisNeuronWeight = nextLayerNeuron.getWeight(neuron_idx+1)
                  totalWkhDk += nextLayerNeuron.errorNode * thisNeuronWeight

                neuron.errorNode = derivate_Ed_To_NETj(layer.type, totalWkhDk, layer.output_array[neuron_idx+1])

        # Delta Weight Update
        prevLayerInputArray = data
        for layerNum in range(len(self.layerArray)):
          layer = self.layerArray[layerNum]

          # Input layer
          if (layer.getType() == 0):
            nextLayerInputArray.extend(data)
            prevLayerInputArray = nextLayerInputArray.copy()

          # Hidden Layer And output
          else:
            prevLayer = self.layerArray[layerNum - 1]
            layer = self.layerArray[layerNum]
            
            for neuron_idx in range(len(layer.getNeuronArray())):
              neuron = layer.getNeuron(neuron_idx)
              accumulative_delta = neuron.getAccumulativeDelta()
              layerNeuron = layer.getNeuron(neuron_idx)

              for layerNeuron_idx in range(len(prevLayer.output_array)):
                if (layerNeuron_idx < len(accumulative_delta)):
                  accumulative_delta[layerNeuron_idx] += self.learning_rate * layerNeuron.errorNode * prevLayer.output_array[layerNeuron_idx]
                else :
                  accumulative_delta.append(self.learning_rate * layerNeuron.errorNode * prevLayer.output_array[layerNeuron_idx])
              neuron.setAccumulativeDelta(accumulative_delta)
              
      self.epoch += 1
      print(self.error)
      print(self.error_threshold)

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

    