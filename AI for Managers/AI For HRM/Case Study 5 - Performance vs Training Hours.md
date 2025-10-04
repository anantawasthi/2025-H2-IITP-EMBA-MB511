Case Study 5: Performance vs Training Hours

### 1. Business Objective

HR wants to check if **employees who undergo more training** also tend to achieve **better performance ratings**.

### 2. Data Science Translation

- Use `PerformanceRating` and `TrainingHours`.

- Perform arithmetic correlation checks.

- Scatter plot to visualize relationship.

### 3. Python Code

```python
"""
Case Study 5: Performance vs Training Hours
-------------------------------------------
Demonstrates:
1. Arithmetic operations (correlation)
2. Scatter plot visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Correlation between training hours and performance
correlation = hr_data['TrainingHours'].corr(hr_data['PerformanceRating'])

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(hr_data['TrainingHours'], hr_data['PerformanceRating'], alpha=0.5, color="purple")
plt.title(f"Training Hours vs Performance Rating (Correlation: {correlation:.2f})")
plt.xlabel("Training Hours")
plt.ylabel("Performance Rating")
plt.show()
```

### 4. Business Output Explanation

- Scatter plot will show if higher training hours **correlate positively** with performance.

- The correlation score (e.g., **+0.35**) indicates the strength of the relationship.

### 5. HR Strategy Use

- Evidence for **investing in employee training**.

- Identifies **highly trained but low-performing employees**, where training may not be effective.

- Helps in **budget justification** for L&D programs.


