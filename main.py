import MBGD
import pandas as pd

df = pd.read_csv("./dataset.csv")
label = df["label"]
data = df.drop(columns=["label"], inplace=False)
label = label.values
data = data.values

mbgd = MBGD.MBGD()
mbgd.setBias(1)

hidden1 = mbgd.createHiddenLayer(2, 2)
output = mbgd.createOutputLayer(2)

mbgd.setLayer([hidden1, output])

mbgd.fit(data, label)

mbgd.printmodel()
print(mbgd.predict(data))