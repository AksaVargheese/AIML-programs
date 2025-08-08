import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("study_hours2.csv")
x=mydata[["no_of_hours"]]
y=mydata[["score"]]
model=knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[8]]))