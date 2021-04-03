import MBGD
import pandas as pd

df = pd.read_csv("./dataset.csv")
label = df["label"]
data = df.drop(columns=["label"], inplace=False)
label = label.values
data = data.values

mbgd = MBGD.MBGD()
mbgd.setBias(1)
mbgd.loadModel(filename="savedmodel.json")
print(mbgd.predict(data))
mbgd.printmodel()