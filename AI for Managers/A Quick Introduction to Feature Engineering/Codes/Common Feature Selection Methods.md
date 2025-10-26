

# **Common Feature Selection Methods**

We’ll begin with **Filter Methods**, which are the simplest and most widely used approach.

---

## **1. Filter Methods**

### **1.1 Introduction**

Filter methods rank features based on statistical measures of their relationship with the target variable, independent of any specific machine learning model. These methods assess relevance through metrics such as correlation, chi-square score, or mutual information. Their model-agnostic nature makes them computationally efficient and suitable for high-dimensional datasets.

---

### **1.2 When to Use**

Use filter methods when the dataset has **a large number of features** and the goal is to quickly remove irrelevant variables before model training. They are ideal for early-stage screening or exploratory feature analysis.

---

### **1.3 When Not to Use**

Avoid filter methods when the model’s feature interactions or non-linear relationships are critical for prediction, as these methods **ignore feature dependencies** and treat each variable independently.

---

### **1.4 Python Implementation (with Docstrings and Explanations)**

```python
"""
Feature Selection using Filter Methods
--------------------------------------
This example demonstrates three common filter techniques:
1. Correlation-based filtering
2. Chi-square test for categorical variables
3. Mutual information for non-linear dependencies

The dataset used is synthetic and small-scale for clarity.
"""

# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif

# Step 1: Create a synthetic dataset
#  - 1000 samples
#  - 8 features (4 informative, 4 redundant)
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=4,
    n_redundant=4,
    random_state=42
)

# Converting to DataFrame for readability
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
df['Target'] = y

# Step 2: Correlation-based filtering
#  - Calculate correlation of each feature with the target variable
corr_values = df.corr(numeric_only=True)['Target'].abs().sort_values(ascending=False)
print("Correlation-based Feature Ranking:")
print(corr_values)

# Step 3: Chi-square test (for categorical-like data)
#  - Discretize continuous data for chi-square application
X_chi = np.abs(X * 10).astype(int)
chi_selector = SelectKBest(score_func=chi2, k=4)
chi_selector.fit(X_chi, y)

print("\nChi-Square Scores:")
for name, score in zip(df.columns[:-1], chi_selector.scores_):
    print(f"{name}: {score:.2f}")

# Step 4: Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores_df = pd.DataFrame({
    'Feature': df.columns[:-1],
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

print("\nMutual Information Scores:")
print(mi_scores_df)
```

**Explanation:**

- The **correlation-based** method quantifies linear association.

- The **chi-square** test checks dependence between categorical variables and the target.

- The **mutual information** approach captures both linear and non-linear dependencies, making it more flexible in complex relationships.

---



# **2. Wrapper Methods**

---

### **2.1 Introduction**

Wrapper methods evaluate subsets of features by actually training and testing a model on them. Instead of ranking features individually, they assess combinations to find the subset that produces the best predictive performance. These methods treat feature selection as a search problem—testing various combinations of features guided by metrics such as accuracy, F1 score, or AUC. Although computationally expensive, wrapper methods often yield more accurate and customized feature subsets.

---

### **2.2 When to Use**

Wrapper methods are ideal when **model performance optimization is more important than computational efficiency**. They work best with moderate-sized datasets and when the feature interactions are complex or non-linear.

---

### **2.3 When Not to Use**

Avoid wrapper methods on **very high-dimensional or large datasets**, as they require repetitive model training that can be computationally prohibitive. They are also less suitable for real-time systems where rapid retraining is necessary.

---

### **2.4 Python Implementation (with Docstrings and Explanations)**

```python
"""
Feature Selection using Wrapper Methods
---------------------------------------
This example demonstrates Recursive Feature Elimination (RFE),
a popular wrapper method that repeatedly fits a model, removes the
least important features, and re-fits until the optimal subset is found.
"""

# Step 1: Import required libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Step 2: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize base model (Random Forest)
model = RandomForestClassifier(random_state=42)

# Step 5: Apply Recursive Feature Elimination (RFE)
#  - n_features_to_select defines how many top features we want to keep
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_train, y_train)

# Step 6: Get selected features
selected_features = X_train.columns[rfe.support_]
print("Selected Features using RFE:")
print(selected_features.tolist())

# Step 7: Evaluate model performance using selected features
model.fit(X_train[selected_features], y_train)
score = model.score(X_test[selected_features], y_test)
print(f"\nModel Accuracy with Selected Features: {score:.3f}")
```

---

### **Explanation**

- **RFE (Recursive Feature Elimination)** removes the least important features iteratively and re-evaluates performance at each step.

- It integrates **model feedback** (feature importance) directly into selection.

- This method adapts to any estimator that provides a `coef_` or `feature_importances_` attribute (e.g., linear models, tree ensembles).

---

### **Example Scenario**

In HR analytics, suppose you’re predicting employee attrition using 40 variables. Wrapper methods can identify which combinations (like *Tenure + Last Appraisal Score + Manager Rating*) yield the highest predictive accuracy, eliminating redundant or weak variables.

---



# **3. Embedded Methods**

---

### **3.1 Introduction**

Embedded methods combine the advantages of **filter and wrapper techniques** by performing feature selection as part of the model training process itself. These methods automatically assign importance scores or penalize less useful features while optimizing the model parameters. As a result, they are **computationally more efficient than wrapper methods** and more **context-aware than simple filters**.

The most common embedded methods include **regularization-based models** like LASSO (L1 penalty), Ridge (L2 penalty), and Elastic Net, as well as **tree-based models** such as Random Forests and Gradient Boosted Trees that internally calculate feature importance. Embedded approaches are especially powerful for large-scale datasets where features are correlated or redundant, as they incorporate both model structure and performance simultaneously during selection.

---

### **3.2 When to Use**

Use embedded methods when the dataset is **moderately large, has correlated features**, or when interpretability and computational efficiency are both desired. They are particularly suitable for **regression and classification problems** where embedded regularization or built-in importance metrics can naturally prune unnecessary features.

---

### **3.3 When Not to Use**

Avoid embedded methods when:

- You require **complete control over feature combinations** or

- When **domain constraints** disallow model-driven automatic feature removal.  
  They may also be unsuitable when explainability requires explicit human reasoning behind feature inclusion rather than algorithmic scoring.

---

### **3.4 Python Implementation (with Detailed Explanation and Docstrings)**

```python
"""
Feature Selection using Embedded Methods
----------------------------------------
This example demonstrates two common embedded methods:
1. LASSO Regression (L1 regularization) — suitable for linear models
2. Random Forest Feature Importance — suitable for non-linear models
"""

# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 2: Load dataset
data = load_boston()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Normalize features (important for LASSO)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: LASSO Regression
lasso = Lasso(alpha=0.05, random_state=42)
lasso.fit(X_train_scaled, y_train)

# Retrieve coefficients (zero coefficients indicate dropped features)
lasso_coefs = pd.Series(lasso.coef_, index=X.columns)
selected_lasso_features = lasso_coefs[lasso_coefs != 0].index.tolist()

print("Selected Features using LASSO Regression:")
print(selected_lasso_features)

# Step 6: Random Forest Feature Importance
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Get feature importance
rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
top_rf_features = rf_importance.sort_values(ascending=False).head(10)

print("\nTop 10 Features by Random Forest Importance:")
print(top_rf_features)

# Step 7: Evaluation (example of usage)
lasso_score = lasso.score(X_test_scaled, y_test)
rf_score = rf.score(X_test, y_test)

print(f"\nModel R² (LASSO): {lasso_score:.3f}")
print(f"Model R² (Random Forest): {rf_score:.3f}")
```

---

### **Explanation**

- **LASSO Regression:**
  
  - Introduces L1 regularization that drives coefficients of less relevant features toward zero.
  
  - Automatically performs variable selection during model fitting.
  
  - Ideal for high-dimensional, sparse data.

- **Random Forest:**
  
  - Uses an ensemble of decision trees to estimate feature importance based on how much each feature reduces impurity across trees.
  
  - Captures both linear and non-linear relationships without explicit scaling.

These embedded methods are efficient, interpretable, and directly integrated into model learning, reducing the need for separate selection steps.

---

### **Example Scenario**

In a **credit risk modeling** problem, a LASSO-based selection can eliminate noisy predictors like *temporary credit utilization* or *one-time promotional usage*, keeping only those that influence default risk. Similarly, a **Random Forest model** can highlight nonlinear predictors such as *income stability ratio* or *age × loan tenure* interactions, guiding data scientists to interpret patterns relevant for underwriting decisions.

---


