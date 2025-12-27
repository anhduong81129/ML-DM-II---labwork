# ML-DM-II---labwork1
# 📊 Exploratory Data Analysis & PCA Visualization

## 📌 Project Overview
This project performs data preprocessing, statistical analysis, correlation analysis, and Principal Component Analysis (PCA) on two real-world datasets:
* **Exam Score Prediction Dataset**
* **Wine Quality Dataset**

**The goal is to:**
* Clean and preprocess data
* Analyze statistical properties (mean & variance)
* Visualize feature correlations
* Reduce dimensionality using PCA
* Visualize data structure in 2D using principal components



## 📁 Project Structure
```text
📂 Project Folder
│
├── plot.py                 # Main analysis & visualization script
├── Exam_Score_Prediction.csv   # Student exam score dataset
├── Wine_Quality.csv            # Wine quality dataset
└── README.md                   # Project documentation


## 🔍 Datasets Description

### 1️⃣ Exam Score Prediction Dataset
This dataset contains student-related features used to analyze factors influencing exam performance.

* **Preprocessing:** Removed `student_id`, encoded categorical variables.
* **Analysis:** Mean/Variance, Correlation Matrices, PCA.
* **Target:** `exam_score`

### 2️⃣ Wine Quality Dataset
This dataset contains physicochemical properties of wine samples.

* **Preprocessing:** Removed missing values, encoded categories.
* **Analysis:** PCA-based dimensionality reduction.
* **Target:** `quality`


## ⚙️ Technologies & Libraries
* **Python 3**
* **Pandas** (Data manipulation)
* **NumPy** (Numerical computation)
* **Matplotlib & Seaborn** (Visualization)
* **Scikit-learn** (StandardScaler, LabelEncoder, PCA)


## ▶️ How to Run the Project

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

### 2. Run the script:
```bash
python plot.py

## 🔍 Analysis Workflow
The analysis is automated using a reusable function `preprocess_and_analyze()` which follows these steps:

### 🔹 1. Data Processing
* **Encoding:** Categorical features are converted to numeric using `LabelEncoder`.
* **Standardization:** Numerical features are scaled using `StandardScaler` to ensure PCA isn't biased by different units.

### 🔹 2. Statistical Analysis
* Automatically computes the **Mean** and **Variance** for every feature in the dataset to understand data distribution.

### 🔹 3. Correlation Analysis
* Generates a **Correlation Matrix** to identify relationships between variables.
* Visualizes these relationships using **Seaborn Heatmaps**.

### 🔹 4. Principal Component Analysis (PCA)
* **Explained Variance:** Calculates how much information each component holds.
* **Scree Plots:** Visualizes the cumulative variance to determine the optimal number of components.
* **Dimensionality Reduction:** Compresses the data into **2 Principal Components**.
* **Visualization:** Generates a 2D scatter plot color-coded by the target variable (`exam_score` or `quality`).



## 🚀 Script Capabilities
The script is designed to automate the following tasks:
- [x] **Statistical Reporting:** Prints mean and variance results directly to the console.
- [x] **Correlation Mapping:** Displays interactive heatmaps for feature relationship analysis.
- [x] **Variance Analysis:** Shows PCA explained variance plots (Scree plots).
- [x] **Dimensionality Visualization:** Generates 2D PCA projections for both datasets.

## 📈 Output Visualizations
The analysis produces three primary visual insights:
1. **Correlation Heatmaps:** Identifies how strongly features relate to the target labels.
2. **Scree Plots:** Shows the cumulative explained variance to justify dimensionality reduction.
3. **2D PCA Scatter Plots:** Clusters data points by target labels (`exam_score` and `quality`) to reveal underlying structures.

[Image of Correlation Heatmap vs PCA Scree Plot vs PCA Scatter Plot]


## 🎯 Learning Outcomes
By completing this analysis, the following core data science skills were applied:
* **Feature Engineering:** Understanding relationships through correlation.
* **Dimensionality Reduction:** Applying PCA to simplify high-dimensional data.
* **Data Visualization:** Communicating complex mathematical transformations effectively.
* **Software Design:** Building a reusable and automated data analysis pipeline.


## 👤 Author
**Ánh Dương**