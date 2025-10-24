#### **Method 1: Normalization (Min–Max Scaling)**

**Introduction:** Rescales numeric features into a fixed range, typically [0, 1], to prevent large-magnitude variables from dominating learning.  
**When to Use:** When features differ widely in scale (e.g., income vs. discount).  
**When *Not* to Use:** When outliers exist or scale carries meaningful information (e.g., logarithmic or power-law distributions).

```python
"""
Normalize numeric variables using Min–Max Scaling.
This ensures that all features contribute equally to model training.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load synthetic dataset
df = pd.read_excel("Retail_Sales_Data.xlsx")

# Initialize scaler
scaler = MinMaxScaler()

# Select numeric columns
num_cols = ['Age', 'Monthly_Income', 'Total_Spend', 'Discount_Used', 'Credit_Score']

# Apply normalization
df_scaled = df.copy()
df_scaled[num_cols] = scaler.fit_transform(df[num_cols])

# Display sample
print(df_scaled.head())

# Interpretation:
# Each numeric value now lies between 0 and 1.
# This helps distance-based models (like KNN or neural nets) converge efficiently.

```

#### **Method 2: Standardization (Z-Score Scaling)**

**Introduction:** Centers features at zero mean and unit variance. It retains outlier sensitivity, useful for normally distributed features.  
**When to Use:** When features approximately follow a Gaussian distribution.  
**When *Not* to Use:** When heavy outliers or skewed data dominate.

```python
"""
Apply Z-score standardization to numeric variables.
This is suitable for algorithms assuming normality (e.g., linear regression, PCA).
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_std = df.copy()
df_std[num_cols] = scaler.fit_transform(df[num_cols])
print(df_std.head())

# Interpretation:
# Each feature now has mean ≈ 0 and standard deviation ≈ 1.
# It improves numerical stability for gradient-based models.

```

#### **Method 3: Binning (Discretization)**

**Introduction:** Converts continuous variables into categorical bins, capturing non-linear effects and reducing noise.  
**When to Use:** When the relationship between the feature and target is non-linear or interpretable in segments (e.g., age groups).  
**When *Not* to Use:** When precise numeric variation is crucial for prediction.

```python
"""
Create age groups by binning continuous Age values into categories.
Helps capture non-linear relationship between Age and churn probability.
"""

bins = [0, 25, 35, 45, 55, 65, 100]
labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
print(df[['Age', 'Age_Group']].head())

# Interpretation:
# Converts numeric range into interpretable categories.
# Useful for tree-based models or business storytelling.

```

#### **Method 4: Log Transformation**

**Introduction:** Reduces right-skewed distributions (e.g., income or spend) by compressing large values.  
**When to Use:** When a variable spans multiple orders of magnitude.  
**When *Not* to Use:** When data contain zeros or negative values.

```python
"""
Apply logarithmic transformation to skewed numeric variables.
This reduces variance and normalizes extreme values.
"""

import numpy as np

df['Log_Total_Spend'] = np.log1p(df['Total_Spend'])  # log(1+x) handles zeros safely
print(df[['Total_Spend', 'Log_Total_Spend']].head())

# Interpretation:
# The transformation stabilizes variance, making distributions more symmetric.

```

#### **Method 5: Polynomial Features**

**Introduction:** Generates interaction or power terms (x², x³) to capture non-linear relationships.  
**When to Use:** When feature–target relation appears curved or multiplicative.  
**When *Not* to Use:** When multicollinearity or overfitting risk is high.

```python
"""
Create polynomial features to model non-linear relationships.
"""

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Monthly_Income', 'Discount_Used']])
poly_cols = poly.get_feature_names_out(['Monthly_Income', 'Discount_Used'])

df_poly = pd.DataFrame(poly_features, columns=poly_cols)
print(df_poly.head())

# Interpretation:
# Adds new derived variables like Monthly_Income^2, Discount_Used^2, and interaction terms.
# Improves flexibility of linear models.

```

#### **Method 6: Ratio and Interaction Features**

**Introduction:** Combines multiple numeric variables to capture meaningful business ratios or relative performance (e.g., Spend / Income).  
**When to Use:** When proportional relationships provide insight (profit margin, utilization ratio).  
**When *Not* to Use:** When denominator values can be zero or unstable.

```python
"""
Create ratio and interaction features to encode relationships between variables.
"""

df['Spend_to_Income_Ratio'] = df['Total_Spend'] / (df['Monthly_Income'] + 1e-6)
df['Avg_Spend_per_Purchase'] = df['Total_Spend'] / (df['Total_Purchases'] + 1e-6)

print(df[['Spend_to_Income_Ratio', 'Avg_Spend_per_Purchase']].head())

# Interpretation:
# These derived variables capture customer spending behavior relative to income and activity level.

```


