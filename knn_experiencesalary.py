import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("knn_experiencesalary.csv")
x=mydata[["experience"]]
y=mydata[["salary"]]
model=knn.KNeighborsRegressor(n_neighbors=2)
model.fit(x,y)
print(model.predict([[4.5]]))