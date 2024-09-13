import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Show the first few rows of the dataset
print(data.head())

print()
# Preprocess the data - for clustering, we will use the Annual Income and
# Spending Score features, as they provide the most relevant information for
# segmenting customers
# Extract relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(X.head())

# We don't need to handle missing values as the dataset is clean
# We will use K-Means Clustering from scikit-learn to group customers into clusters
# One of the important steps is deciding the number of clusters
# We can use the Elbow Method to find the optimal number of clusters by
# plotting the within-cluster sum of squares (WCSS) for different values of k
# Elbow method to find optimal number of clusters (k)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)    # Inertia is the WCSS

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()  # We will use k = 5 clusters based on the plot

# Apply K-Means clustering with the chosen number of clusters, k = 5
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster labels to the original data
data['Cluster'] = y_kmeans

# We can visualize the customer segments in a 2D plot using Annual Income and Spending Score as axes
plt.figure(figsize=(10, 7))
plt.scatter(X.values[y_kmeans == 0, 0], X.values[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X.values[y_kmeans == 1, 0], X.values[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X.values[y_kmeans == 2, 0], X.values[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X.values[y_kmeans == 3, 0], X.values[y_kmeans == 3, 1], s=100, c='purple', label='Cluster 4')
plt.scatter(X.values[y_kmeans == 4, 0], X.values[y_kmeans == 4, 1], s=100, c='orange', label='Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

# This scatter plot will show the customers segmented into different colors based on their income
# and spending behavior, with the centroids marked in yellow
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# We can evaluate the performance of the clustering algorithm using Silhouette Score
# which measures how similar a point is to its own cluster compared to other clusters
score = silhouette_score(X, y_kmeans)
print()
print(f'Silhouette Score: {score}')
