import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

# =========================================================
# Load datasets

base_path = os.path.dirname(os.path.abspath(__file__))

red_wine = pd.read_csv(os.path.join(base_path,'winequality-red.csv'), sep=';')
white_wine = pd.read_csv(os.path.join(base_path,'winequality-white.csv'), sep=';')

# =========================================================
# SVM Classification and Analysis
def run_svm_analysis(df, name):
    print(f"\n==============================")
    print(f"SVM Analysis for {name}")
    print(f"==============================")

    # Prepare data
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =====================================================
    # Normalize features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =====================================================
    # 1. Dataset analysis — Linear vs Nonlinear separability
    # Reduce to 2D using PCA for visualization
    pca_2d = PCA(n_components=2)
    X_vis = pca_2d.fit_transform(X_train_scaled)

    plt.figure()
    scatter = plt.scatter(X_vis[:,0], X_vis[:,1], c=y_train, cmap='viridis', s=15)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"2D PCA Projection of {name}")
    plt.colorbar(scatter, label="Quality Label")
    plt.show()

    print("→ From the PCA plot, classes are overlapping → data is NOT linearly separable.")

    # =====================================================
    # 2. Train Linear SVM
    svm_linear = SVC(kernel='linear', C=1)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred_linear = svm_linear.predict(X_test_scaled)
    acc_linear = accuracy_score(y_test, y_pred_linear)

    print(f"\nLinear SVM Accuracy: {acc_linear:.4f}")

    # =====================================================
    # 3. Train RBF (non-linear) SVM
    svm_rbf = SVC(kernel='rbf', C=10, gamma='scale')
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred_rbf = svm_rbf.predict(X_test_scaled)
    acc_rbf = accuracy_score(y_test, y_pred_rbf)

    print(f"RBF (Non-linear) SVM Accuracy: {acc_rbf:.4f}")

    # =====================================================
    # 4. Cross-validation performance
    scores = cross_val_score(svm_rbf, X_train_scaled, y_train, cv=5)
    print(f"5-Fold CV Accuracy (RBF SVM): {scores.mean():.4f}")

    # =====================================================
    # 5. Classification report
    print("\nClassification Report (RBF SVM):")
    print(classification_report(y_test, y_pred_rbf))

    # =====================================================
    # 6. Multi-class handling explanation
    print("SVM handles multi-class classification using One-vs-One strategy by default in sklearn.")


# =========================================================
# Run SVM on both datasets

run_svm_analysis(red_wine, "Red Wine Dataset")
run_svm_analysis(white_wine, "White Wine Dataset")
