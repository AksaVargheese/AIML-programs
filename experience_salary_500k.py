import pandas as pd
import matplotlib.pyplot as pyplot
import sklearn.linear_model as lm
mydata=pd.read_csv("experience_salary_500k.csv")
x=mydata[["years_of_experience"]]
y=mydata[["salary"]]
pyplot.scatter(x,y)
pyplot.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[4.5]]))