# Antarctic Penguin Species Clustering 🐧

**Unsupervised Machine Learning Project | Data Science Portfolio**

![Penguin Species](https://imgur.com/orZWHly.png)
*Source: @allison_horst <https://github.com/allisonhorst/penguins>*

## 📋 Project Overview

This project applies **unsupervised machine learning techniques** to identify distinct penguin species groups from morphological measurements collected in Antarctica. Using K-Means clustering, I successfully identified **3 distinct clusters** corresponding to the three known penguin species: **Adelie**, **Chinstrap**, and **Gentoo**.

## 🎯 Learning Objectives & Achievements

- ✅ **Data Preprocessing**: Successfully handled categorical variables and applied standardization
- ✅ **Unsupervised Learning**: Implemented K-Means clustering algorithm
- ✅ **Model Optimization**: Used Elbow Method to determine optimal number of clusters (k=3)
- ✅ **Dimensionality Analysis**: Applied PCA for feature importance analysis
- ✅ **Data Visualization**: Created both static (Matplotlib) and interactive (Plotly) 3D visualizations
- ✅ **Interactive Analysis**: Implemented Plotly for dynamic cluster exploration
- ✅ **Feature Engineering**: Converted categorical sex variable to numerical format

## 📊 Dataset Information

**Source**: Palmer Station, Antarctica LTER (Long Term Ecological Research Network)  
**Collected by**: Dr. Kristen Gorman

| Feature | Description | Data Type |
|---------|-------------|-----------|
| `culmen_length_mm` | Culmen length (mm) | Continuous |
| `culmen_depth_mm` | Culmen depth (mm) | Continuous |
| `flipper_length_mm` | Flipper length (mm) | Continuous |
| `body_mass_g` | Body mass (g) | Continuous |
| `sex` | Penguin sex | Categorical → Encoded |

**Total Features**: 5  
**Target Variable**: Species (unlabeled - to be discovered through clustering)

## 🔬 Methodology

### 1. Data Preprocessing

- **Categorical Encoding**: Converted `sex` variable using category codes
- **Feature Scaling**: Applied StandardScaler to normalize all numerical features
- **Data Structure**: Maintained original dataframe while creating scaled version for modeling

### 2. Optimal Cluster Selection

- **Method**: Elbow Method
- **Range Tested**: k = 1 to 10 clusters
- **Result**: Optimal k = 3 (matches expected number of penguin species)
- **Visualization**: Clear elbow pattern observed in inertia plot + PCA feature inertia

### 3. Clustering Implementation

- **Algorithm**: K-Means Clustering
- **Parameters**:
  - n_clusters = 3
  - random_state = 42 (for reproducibility)
- **Feature Set**: All 5 preprocessed features

### 4. Feature Importance Analysis

- **Method**: Analysis of cluster center coordinates
- **Approach**: Calculated absolute mean of cluster centers across all features
- **Result**: Ranked features by their contribution to cluster separation

### 5. Dimensionality Analysis

- **Technique**: Principal Component Analysis (PCA)
- **Purpose**: Understand feature variance contribution
- **Visualization**: Bar chart showing explained variance per component

## 📈 Results & Insights

### Clustering Performance

- **Number of Clusters Identified**: 3
- **Cluster Distribution**: Successfully separated penguins into distinct groups
- **Visual Validation**: 3D scatter plot shows clear cluster separation

### Feature Importance Ranking

Based on cluster center analysis, features are ranked by their importance in distinguishing penguin groups:

1. **Most Important Features**: [Determined from cluster analysis]
2. **Moderate Importance**: [Secondary discriminating features]
3. **Lower Importance**: [Supporting features]

### Visualization Insights

- **Static 3D Plot (Matplotlib)**: Professional publication-ready visualization
- **Interactive 3D Plot (Plotly)**: Dynamic exploration with rotation, zoom, and hover details
- **Axes Used**: Flipper Length, Culmen Depth, Culmen Length  
- **Color Coding**: Each cluster represented with distinct colors
- **Pattern**: Clear geometric separation between the three species groups
- **Interactivity**: Toggle clusters, explore data points, and examine cluster boundaries

## 🛠️ Technical Implementation

### Libraries & Tools Used

```python
# Data Manipulation & Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
```

### Key Code Implementations

- **Data Scaling Pipeline**: StandardScaler for feature normalization
- **Elbow Method Loop**: Automated optimal k selection
- **Static 3D Visualization**: Custom matplotlib 3D scatter plot
- **Interactive 3D Visualization**: Plotly 3D scatter plot with hover functionality
- **Feature Analysis**: Cluster center interpretation

## 📁 Project Structure

```
DS_project10(UnsupervisedML)/
├── notebook.ipynb          # Main analysis notebook
├── penguins.csv           # Dataset
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── 3d_scatter_plot.png    # Generated visualization
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Skills Demonstrated

### Data Science Skills

- **Exploratory Data Analysis (EDA)**
- **Feature Engineering & Preprocessing**
- **Unsupervised Machine Learning**
- **Model Validation & Optimization**
- **Statistical Analysis**

### Technical Skills

- **Python Programming**
- **Pandas for Data Manipulation**
- **Scikit-learn for Machine Learning**
- **Matplotlib/Seaborn for Static Visualization**
- **Plotly for Interactive Visualization**
- **NumPy for Numerical Computing**

### Analytical Skills

- **Pattern Recognition**
- **Statistical Interpretation**
- **Data-Driven Decision Making**
- **Scientific Method Application**

## 🎓 Learning Outcomes

This project enhanced my understanding of:

- **Unsupervised learning principles** and when to apply them
- **Clustering algorithms** and their parameter optimization
- **Feature scaling importance** in distance-based algorithms
- **Dimensionality reduction techniques** (PCA)
- **Scientific data analysis workflow**
- **Biological data interpretation** in ecological research context

## 🔮 Future Improvements

- [ ] **Advanced Clustering**: Compare with hierarchical clustering, DBSCAN
- [x] **Interactive Visualizations**: ✅ Implemented Plotly for enhanced exploration  
- [ ] **Statistical Validation**: Add silhouette analysis for cluster quality
- [ ] **Feature Selection**: Implement automated feature importance selection
- [ ] **Model Comparison**: Benchmark against other unsupervised algorithms
- [ ] **Dashboard Creation**: Build interactive Plotly Dash application

## 📚 References & Acknowledgments

- **Data Source**: Palmer Station, Antarctica LTER
- **Original Collector**: Dr. Kristen Gorman
- **Artwork Credit**: @allison_horst
- **Dataset Repository**: <https://github.com/allisonhorst/penguins>

---

**Portfolio Project** | *Demonstrating Unsupervised Machine Learning Proficiency*  
**Status**: ✅ Completed | **Level**: Intermediate | **Focus**: Clustering & Dimensionality Analysis
