import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score

import os

#===============================================================
# 1. DATA LOADING (red wine and white wine dataset)

base_path = os.path.dirname(os.path.abspath(__file__))

red_wine_path = os.path.join(base_path, 'winequality-red.csv')
white_wine_path = os.path.join(base_path, 'winequality-white.csv')

red_wine = pd.read_csv(red_wine_path, sep=';')         # Using sep=';' as the CSV uses semicolons as delimiters (it mean that can slice the data correctly)
white_wine = pd.read_csv(white_wine_path, sep=';')


#===============================================================
# 2. CLUSTERING EXPERIMENTS (using K-means)
def run_clustering_experiments(df, title):
    """
    Experimental Protocol:
    - Separate features from labels (quality).
    - Standardize features.
    - Loop through k=2 to 10.
    - Calculate internal (SSE, Silhouette) and external (NMI) metrics.
    """
    # Prepare features (X) and ground truth labels (y)
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    # Feature Scaling (Crucial for K-means distance calculations)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ks = range(2, 11) #if k=1 , all points belong to the same cluster, so no clustering quality can be MEASURED
    sse = []
    silhouettes = []
    nmis = []
    
    for k in ks:
        # Centroid initialization: 'k-means++'
        # n_init=10: Run the algorithm 10 times with different seeds to find the best local minimum
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate Quality Criteria
        sse.append(kmeans.inertia_)  # Sum of Squared Errors
        silhouettes.append(silhouette_score(X_scaled, clusters))
        nmis.append(normalized_mutual_info_score(y, clusters))
    
    return ks, sse, silhouettes, nmis

# Execute experiments
ks_red, sse_red, sil_red, nmi_red = run_clustering_experiments(red_wine, "Red Wine")
ks_white, sse_white, sil_white, nmi_white = run_clustering_experiments(white_wine, "White Wine")

#===============================================================
# 3. PLOTTING RESULTS

# Figure 1: Elbow Method (SSE)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ks_red, sse_red, marker='o', color='tab:red')
plt.title('Red Wine: Elbow Method (SSE)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')

plt.subplot(1, 2, 2)
plt.plot(ks_white, sse_white, marker='o', color='tab:blue')
plt.title('White Wine: Elbow Method (SSE)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.tight_layout()
plt.show() # Shows first figure

# Figure 2: Silhouette Score
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ks_red, sil_red, marker='o', color='tab:red')
plt.title('Red Wine: Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 2, 2)
plt.plot(ks_white, sil_white, marker='o', color='tab:blue')
plt.title('White Wine: Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show() # Shows second figure

# Figure 3: NMI (External Validation)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ks_red, nmi_red, marker='o', color='tab:red')
plt.title('Red Wine: NMI (vs Quality)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('NMI')

plt.subplot(1, 2, 2)
plt.plot(ks_white, nmi_white, marker='o', color='tab:blue')
plt.title('White Wine: NMI (vs Quality)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('NMI')
plt.tight_layout()
plt.show() # Shows third figure

# Display quality metrics for a specific k
print(f"Red Wine Quality (k=6): SSE={sse_red[4]:.2f}, Silhouette={sil_red[4]:.3f}, NMI={nmi_red[4]:.3f}")
print(f"White Wine Quality (k=6): SSE={sse_white[4]:.2f}, Silhouette={sil_white[4]:.3f}, NMI={nmi_white[4]:.3f}")