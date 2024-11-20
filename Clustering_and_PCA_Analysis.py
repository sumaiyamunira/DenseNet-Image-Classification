import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# File paths
file_path_val = 'val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
file_path_v2 = 'v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'

# Define chunk size
chunk_size = 1000

# Initialize scaler and PCA
scaler = StandardScaler()
pca = PCA(n_components=2)  # Reduce to 2D for visualization

# Process first dataset (val_eva02) incrementally
val_scaled_data = []

for chunk in pd.read_csv(file_path_val, chunksize=chunk_size):
    # Select only numeric columns
    chunk = chunk.select_dtypes(include=[np.number])

    # Handle missing values by filling with mean
    chunk = chunk.fillna(chunk.mean())

    # Scale chunk and store the result
    scaled_chunk = scaler.fit_transform(chunk)
    val_scaled_data.append(scaled_chunk)

# Concatenate all processed chunks for val_eva02
val_features_scaled = np.concatenate(val_scaled_data, axis=0)

# Process second dataset (v2_eva02) incrementally
v2_scaled_data = []

for chunk in pd.read_csv(file_path_v2, chunksize=chunk_size):
    # Select only numeric columns
    chunk = chunk.select_dtypes(include=[np.number])

    # Handle missing values by filling with mean
    chunk = chunk.fillna(chunk.mean())

    # Scale chunk and store the result
    scaled_chunk = scaler.transform(chunk)  # Use transform to keep the same scaling
    v2_scaled_data.append(scaled_chunk)

# Concatenate all processed chunks for v2_eva02
v2_features_scaled = np.concatenate(v2_scaled_data, axis=0)

# Apply PCA for dimensionality reduction
val_pca = pca.fit_transform(val_features_scaled)
v2_pca = pca.transform(v2_features_scaled)

# Visualize PCA results
plt.scatter(val_pca[:, 0], val_pca[:, 1], alpha=0.5, label='val_eva02')
plt.scatter(v2_pca[:, 0], v2_pca[:, 1], alpha=0.5, label='v2_eva02')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('PCA of Pre-Trained Features for Two Test Sets')
plt.show()

# Select a Clustering Algorithm
kmeans = KMeans(n_clusters=5, random_state=42)  # Choose number of clusters based on experimentation
val_clusters = kmeans.fit_predict(val_features_scaled)
v2_clusters = kmeans.predict(v2_features_scaled)

# Cluster Visualization
plt.scatter(val_pca[:, 0], val_pca[:, 1], c=val_clusters, cmap='viridis', label='val_eva02 clusters')
plt.scatter(v2_pca[:, 0], v2_pca[:, 1], c=v2_clusters, cmap='plasma', label='v2_eva02 clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Cluster Analysis on Pre-Trained Features')
plt.show()

# Similarity Analysis
similarity_matrix = cosine_similarity(val_features_scaled, v2_features_scaled)
avg_similarity = np.mean(similarity_matrix)
print("Average Cosine Similarity between val_eva02 and v2_eva02:", avg_similarity)
