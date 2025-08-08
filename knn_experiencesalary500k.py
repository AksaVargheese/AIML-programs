import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("experience_salary_500k.csv")
x=mydata[["years_of_experience"]]
y=mydata[["salary"]]
model=knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[4]]))