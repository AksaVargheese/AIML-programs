import pandas as pd 
import matplotlib.pyplot as pyp
import sklearn.linear_model as lm
mydata=pd.read_csv("height_weight_data.csv")
x=mydata[["height"]]
y=mydata[["weight"]]
pyp.scatter(x,y)
pyp.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[160]]))