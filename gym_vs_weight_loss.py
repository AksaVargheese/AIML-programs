import pandas as pd 
import matplotlib.pyplot as pyp
import sklearn.linear_model as lm
mydata=pd.read_csv("gym_vs_weight_loss.csv")
x=mydata[["weekly_gym_hours"]]
y=mydata[["weight_loss_kg"]]
pyp.scatter(x,y)
pyp.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[4]]))