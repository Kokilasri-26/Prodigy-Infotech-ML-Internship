# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 2. Load dataset
data = pd.read_csv("Mall_Customers.csv")

# 3. Select features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Elbow Method to find optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 5. Plot Elbow graph
plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 6. Train K-Means with optimal clusters (usually 5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 7. Add cluster labels to dataset
data["Cluster"] = y_kmeans

# 8. Visualize clusters
plt.figure()
plt.scatter(X[y_kmeans == 0]["Annual Income (k$)"], 
            X[y_kmeans == 0]["Spending Score (1-100)"])
plt.scatter(X[y_kmeans == 1]["Annual Income (k$)"], 
            X[y_kmeans == 1]["Spending Score (1-100)"])
plt.scatter(X[y_kmeans == 2]["Annual Income (k$)"], 
            X[y_kmeans == 2]["Spending Score (1-100)"])
plt.scatter(X[y_kmeans == 3]["Annual Income (k$)"], 
            X[y_kmeans == 3]["Spending Score (1-100)"])
plt.scatter(X[y_kmeans == 4]["Annual Income (k$)"], 
            X[y_kmeans == 4]["Spending Score (1-100)"])

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            marker="X", s=200)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()
