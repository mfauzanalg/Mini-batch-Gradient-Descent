from utils import getErrorNodeOutput, arrayMultiplication, derivate_Ed_To_NETj, unique


def confusion_matrix(y_test, pred):
    # TN, TP, FN, FP
    unique_val = unique(y_test)
    confusion_matrix = []

    for i in range(len(unique_val)):
      confusion_matrix.append([0] * len(unique_val))

    for y_idx in range(len(y_test)):
      unique_index = unique_val.index(y_test[y_idx])

      # Jika prediksinya benar
      if (pred[y_idx] == y_test[y_idx]):
        confusion_matrix[unique_index][unique_index] += 1
 
      # Jika salah prediksi
      else:
        pred_index = unique_val.index(pred[y_idx])
        confusion_matrix[unique_index][pred_index] += 1

    return confusion_matrix
  
def TP(c_matrix, index):
  return c_matrix[index][index]

def TN(c_matrix, index):
   count = 0
   dim = len(c_matrix)
   for i in range(dim):
     for j in range(dim):
       if (i != index or j != index):
         count += 1
   return count 

def FP(c_matrix, index):
   count = 0
   dim = len(c_matrix)
   for i in range(dim):
     if(i != index):
      count += c_matrix[index][i]
   return count 

def FN(c_matrix, index):
  count = 0
  dim = len(c_matrix)
  for i in range(dim):
    if(i != index):
      count += c_matrix[i][index]
  return count  

def precision(TP, FP):
  return (TP/(TP+FP))

def recall(TP, FN):
  return (TP/(TP+FN))

def F1(TP, FP, FN):
  return (2*TP/(2*TP + FP + FN))

def accuracy(c_matrix):
  acc = 0
  count = 0
  dim = len(c_matrix)

  for i in range(dim):
    for j in range(dim):
      count += c_matrix[i][j]

  for i in range(dim):
    acc += c_matrix[i][i]

  return acc/count

def classification_report(c_matrix):
  dim = len(c_matrix)
  acc = 0
  for i in range(len(c_matrix)):
    print("Class ", i)
    print("Precission : ", precision(TP(c_matrix, i), FP(c_matrix, i)))
    print("Recall     : ", recall(TP(c_matrix, i), FN(c_matrix, i)))
    print("F1         : ", F1(TP(c_matrix, i), FP(c_matrix, i), FN(c_matrix, i)))
  print()
  print("Accuracy   :", accuracy(c_matrix))
  return "finish"

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confuse = confusion_matrix(y_true, y_pred)

confuse = [[7,8,9], [1,2,3], [3,2,1]]
print(confuse)
print(classification_report(confuse))