import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import os

#=========================================================
# 1. LOADING DATASET

base_path = os.path.dirname(os.path.abspath(__file__))

red_wine_path = os.path.join(base_path, 'winequality-red.csv')
white_wine_path = os.path.join(base_path, 'winequality-white.csv')

red_wine = pd.read_csv(red_wine_path, sep=';')
white_wine = pd.read_csv(white_wine_path, sep=';')

#=========================================================
# 2. K-NN CLASSIFICATION AND ANALYSIS
def run_knn_analysis(df, name):
    print(f"\n--- Analysis for {name} ---")
    
    #================================
    # Prepare data
    X = df.drop('quality', axis=1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #================================
    # Run k-nn and calculate error
    knn = KNeighborsClassifier(n_neighbors=5) #k=5
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    print(f"Initial k-NN (k=5) Classification Error: {error:.4f}")

    #================================
    # Vary the value of k
    for k in [1, 3, 7, 11]:
        knn_v = KNeighborsClassifier(n_neighbors=k)
        knn_v.fit(X_train, y_train)
        err = 1 - accuracy_score(y_test, knn_v.predict(X_test))
        print(f"k={k} Error: {err:.4f}")

    #================================
    # Normalize the dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn.fit(X_train_scaled, y_train)
    error_scaled = 1 - accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"Error after Normalization: {error_scaled:.4f}")


    #================================
    # Plot Error vs. k
    k_values = range(1, 16)
    errors = []

    for k in k_values:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_scaled, y_train)
        err = 1 - accuracy_score(y_test, knn_k.predict(X_test_scaled))
        errors.append(err)

    plt.figure()
    plt.plot(k_values, errors, marker='o')
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Classification Error")
    plt.title(f"Error vs k for {name}")
    plt.show()

    #================================
    # PCA and SVD Projection
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_train_scaled)
    knn.fit(X_pca, y_train)
    error_pca = 1 - accuracy_score(y_test, knn.predict(pca.transform(X_test_scaled)))
    print(f"Error with PCA (5 components): {error_pca:.4f}")

    # Plot Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA Explained Variance for {name}")
    plt.show()

    svd = TruncatedSVD(n_components=5)
    X_svd = svd.fit_transform(X_train_scaled)
    knn.fit(X_svd, y_train)
    error_svd = 1 - accuracy_score(y_test, knn.predict(svd.transform(X_test_scaled)))
    print(f"Error with SVD (5 components): {error_svd:.4f}")

    #================================
    # K-cross validation
    print("\n5-Fold Cross-Validation for different k:")
    best_k = 1
    best_score = 0

    for k in range(1, 16):
        knn_cv = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_cv, X_train_scaled, y_train, cv=5)
        mean_score = scores.mean()
        print(f"k={k:2d} -> CV Accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"\nBest k from Cross-Validation: k={best_k} with Accuracy={best_score:.4f}")

    #================================
    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    loo_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train, cv=loo)
    print(f"Leave-One-Out Accuracy: {loo_scores.mean():.4f}")
    print(f"Leave-One-Out Error: {1 - loo_scores.mean():.4f}")

#=========================================================

# Execute for both datasets
run_knn_analysis(red_wine, "Red Wine")
run_knn_analysis(white_wine, "White Wine")
