import pandas as pd 
import matplotlib.pyplot as pyp
import sklearn.linear_model as lm
mydata=pd.read_csv("mobile.csv")
x=mydata[["mobile usage(hours)"]]
y=mydata[["battery left"]]
pyp.scatter(x,y)
pyp.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[8]]))