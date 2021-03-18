import FFNN
import pandas as pd

df = pd.read_csv("./dataset.csv")
label = df["label"]
data = df.drop(columns=["label"], inplace=False)
label = label.values
data = data.values

ffnn = FFNN.FFNN()
ffnn.setBias(1)
ffnn.loadModel(filename="model.json")
print(ffnn.predict(data))
ffnn.printmodel()