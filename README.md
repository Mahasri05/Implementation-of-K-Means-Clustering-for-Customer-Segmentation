# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and mataplotlib.pyplot
2.Read the dataset and transform it 
3.Import KMeans and fit the data in the model 
4. Plot the cluster graph
   

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Mahasri D 
RegisterNumber:24901210 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
print(y_pred)
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)
![Screenshot 2024-12-15 212443](https://github.com/user-attachments/assets/c5ab84cb-b391-41e0-a69a-83d39f0c66f6)

![Screenshot 2024-12-15 212534](https://github.com/user-attachments/assets/2ba1fb82-3892-472c-9f20-167c5f8bef83)

![Screenshot 2024-12-15 212521](https://github.com/user-attachments/assets/bcbe0cf8-c14f-4c7a-90d0-a1e1e0d1a112)

![Screenshot 2024-12-15 212542](https://github.com/user-attachments/assets/439ab15c-c666-453d-9b32-f5c6bb274ba3)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
