import pandas as pd
import matplotlib.pyplot as pyplot
import sklearn.linear_model as lm
mydata=pd.read_csv("study_hours_vs_scores.csv")
x=mydata[["no_of_hours"]]
y=mydata[["score"]]
pyplot.scatter(x,y)
pyplot.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[4.5]]))