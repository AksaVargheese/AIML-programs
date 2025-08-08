import pandas as pd 
import matplotlib.pyplot as pyp
import sklearn.linear_model as lm
mydata=pd.read_csv("study_hours2.csv")
x=mydata[["no_of_hours"]]
y=mydata[["score"]]
pyp.scatter(x,y)
pyp.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[8]]))