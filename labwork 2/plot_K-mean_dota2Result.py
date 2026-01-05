import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

# 1. Setup and Preprocessing
train_df = pd.read_csv('dota2Train.csv', header=None)
sample_df = train_df.sample(n=5000, random_state=42) # Sampling for performance

y = sample_df[0]  # Target (Win/Loss)
X = sample_df.drop(columns=[0])

# Standardization is crucial for K-means (distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Define Evaluation Function
def evaluate_clustering(data, true_labels, n_clusters=2):
    start_time = time.time()
    # Initialize and fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(data)
    elapsed_time = time.time() - start_time
    
    # Calculate Metrics
    sil = silhouette_score(data, cluster_labels) # Density/Separation
    ari = adjusted_rand_score(true_labels, cluster_labels) # Match with ground truth
    
    return sil, ari, elapsed_time, cluster_labels

# --- Scenario A: Clustering on Original Data (116 Features) ---
sil_orig, ari_orig, time_orig, _ = evaluate_clustering(X_scaled, y)

# --- Scenario B: Clustering after PCA (2D) ---
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
sil_pca, ari_pca, time_pca, labels_pca = evaluate_clustering(X_pca_2d, y)

# --- Scenario C: Clustering on a Random Subspace ---
np.random.seed(42)
# Select 20 random feature indices
random_indices = np.random.choice(range(X_scaled.shape[1]), size=20, replace=False)
X_random = X_scaled[:, random_indices]
sil_rand, ari_rand, time_rand, _ = evaluate_clustering(X_random, y)

# 3. Print Results
print(f"Original (116D) - Silhouette: {sil_orig:.4f}, ARI: {ari_orig:.4f}")
print(f"PCA (2D)        - Silhouette: {sil_pca:.4f}, ARI: {ari_pca:.4f}")
print(f"Random (20D)    - Silhouette: {sil_rand:.4f}, ARI: {ari_rand:.4f}")