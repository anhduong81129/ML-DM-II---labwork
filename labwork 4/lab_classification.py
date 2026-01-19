import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# =====================
# Load datasets
# =====================
base_path = os.path.dirname(os.path.abspath(__file__))

red = pd.read_csv(os.path.join(base_path, "winequality-red.csv"), sep=";")
white = pd.read_csv(os.path.join(base_path, "winequality-white.csv"), sep=";")

# =====================
# Prepare data
# =====================
def prepare_data(data):
    X = data.drop("quality", axis=1)
    y = (data["quality"] >= 6).astype(int)
    return X, y

X_red, y_red = prepare_data(red)
X_white, y_white = prepare_data(white)

# Create output folder
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# =====================
# 1. Class Distribution Plots
# =====================
def plot_class_distribution(y, title, filename):
    counts = y.value_counts().sort_index()
    plt.figure()
    plt.bar(["Bad (0)", "Good (1)"], counts)
    plt.title(title)
    plt.ylabel("Number of Samples")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_class_distribution(y_red, "Red Wine Class Distribution", "red_class_distribution.png")
plot_class_distribution(y_white, "White Wine Class Distribution", "white_class_distribution.png")

# =====================
# Train models once
# =====================
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return y_test, dt.predict(X_test), rf.predict(X_test)

ytest_red, dt_red, rf_red = train_models(X_red, y_red)
ytest_white, dt_white, rf_white = train_models(X_white, y_white)

# =====================
# 2. Accuracy Comparison Plots
# =====================
def plot_accuracy(y_test, dt_pred, rf_pred, title, filename):
    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    plt.figure()
    plt.bar(["Decision Tree", "Random Forest"], [dt_acc, rf_acc])
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_accuracy(ytest_red, dt_red, rf_red, "Accuracy - Red Wine", "accuracy_red.png")
plot_accuracy(ytest_white, dt_white, rf_white, "Accuracy - White Wine", "accuracy_white.png")

# =====================
# 3. Confusion Matrix Plots
# =====================
def plot_confusion(cm, title, filename):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_confusion(confusion_matrix(ytest_red, dt_red),
               "DT Confusion Matrix - Red Wine", "cm_dt_red.png")

plot_confusion(confusion_matrix(ytest_red, rf_red),
               "RF Confusion Matrix - Red Wine", "cm_rf_red.png")

plot_confusion(confusion_matrix(ytest_white, dt_white),
               "DT Confusion Matrix - White Wine", "cm_dt_white.png")

plot_confusion(confusion_matrix(ytest_white, rf_white),
               "RF Confusion Matrix - White Wine", "cm_rf_white.png")

print("All figures generated in the 'figures' folder.")