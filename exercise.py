import pandas as pd 
import matplotlib.pyplot as pyp
import sklearn.linear_model as lm
mydata=pd.read_csv("exercise.csv")
x=mydata[["exercise time"]]
y=mydata[["calories burnt"]]
pyp.scatter(x,y)
pyp.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[60]]))