Case Study 3: Workforce Demographics Report

### 1. Business Objective

The HR leadership team needs a **clear snapshot of workforce demographics** — gender balance, age distribution, and marital status. This helps in diversity, equity, and inclusion (DEI) initiatives and workforce planning.

### 2. Data Science Translation

- Convert data types (e.g., ensure categorical columns are consistent).

- Rename columns into business-friendly names.

- Sort employees by **Age**.

- Generate **summary tables** and a **pie chart** for gender ratio.

### 3. Python Code

```python
"""
Case Study 3: Workforce Demographics Report
-------------------------------------------
Demonstrates:
1. Data type conversion (object → category)
2. Renaming columns
3. Sorting data
4. Simple aggregation and visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "/mnt/data/MB511_HR_Dataset.xlsx"
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Convert categorical fields to 'category' type
for col in ['Gender', 'MaritalStatus', 'Department']:
    hr_data[col] = hr_data[col].astype('category')

# Rename columns for reporting
hr_data = hr_data.rename(columns={
    "Gender": "EmployeeGender",
    "MaritalStatus": "EmployeeMaritalStatus"
})

# Sort employees by Age
sorted_data = hr_data.sort_values(by="Age", ascending=True)

# Gender distribution
gender_dist = hr_data['EmployeeGender'].value_counts()

# Visualization
plt.figure(figsize=(6,6))
gender_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=["#66b3ff","#ff9999"])
plt.ylabel("")
plt.title("Gender Distribution of Workforce")
plt.show()
```

### 4. Business Output Explanation

- A **sorted employee list** by age (useful for retirement and succession planning).

- A **pie chart** showing male vs female workforce share.

- HR can also break this down further by marital status or department.

### 5. HR Strategy Use

- Ensuring **gender diversity** targets are met.

- Identifying **age clusters** for succession planning.

- Supporting **employee benefits design** (e.g., young workforce → learning programs, older workforce → health & retirement plans).

---
