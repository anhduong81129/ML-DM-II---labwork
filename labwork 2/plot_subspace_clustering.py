import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Column 0 is the target (Win/Loss), others are features
train_df = pd.read_csv('dota2Train.csv', header=None)

# 2. Sample the data for faster visualization
sample_df = train_df.sample(n=10000, random_state=42)

# 3. Separate features and target
y = sample_df[0]
X = sample_df.drop(columns=[0])

# 4. Standardize the features
# PCA is sensitive to the scale of features, so standardization is essential
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Perform PCA for 2D Visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# Plot 2D PCA
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
plt.title('2D PCA of Dota 2 Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, ticks=[-1, 1], label='Win (-1) / Loss (1)')
plt.savefig('pca_2d.png')
plt.show()

# 6. Perform PCA for 3D Visualization
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Plot 3D PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='viridis', alpha=0.5, s=10)
ax.set_title('3D PCA of Dota 2 Data')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
fig.colorbar(scatter, ax=ax, ticks=[-1, 1], label='Win (-1) / Loss (1)')
plt.savefig('pca_3d.png')
plt.show()

# Print Explained Variance
print("Explained variance ratio (2D):", pca_2d.explained_variance_ratio_)
print("Explained variance ratio (3D):", pca_3d.explained_variance_ratio_)