Case Study 10: Building an HR Dashboard

### 1. Business Objective

The CHRO wants a **quick one-page HR dashboard** summarizing **compensation, attrition, and satisfaction** in one view.

### 2. Data Science Translation

- Combine multiple aggregations:
  
  - Avg Salary by Department
  
  - Attrition Rate by Department
  
  - Job Satisfaction by Work Mode

- Visualize with **subplots**.

### 3. Python Code

```python
"""
Case Study 10: HR Dashboard
---------------------------
Demonstrates:
1. Combining multiple aggregations
2. Creating subplot dashboard with Matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Aggregations
salary_dept = hr_data.groupby("Department")['MonthlySalaryINR'].mean()
attrition_dept = hr_data.groupby("Department")['Attrition'].mean()*100
satisfaction_mode = hr_data.groupby("WorkMode")['JobSatisfaction'].mean()

# Dashboard visualization
fig, axes = plt.subplots(1,3, figsize=(18,5))

# Salary chart
salary_dept.plot(kind='bar', ax=axes[0], color="skyblue")
axes[0].set_title("Avg Salary by Dept")

# Attrition chart
attrition_dept.plot(kind='bar', ax=axes[1], color="salmon")
axes[1].set_title("Attrition Rate by Dept (%)")

# Satisfaction chart
satisfaction_mode.plot(kind='bar', ax=axes[2], color="lightgreen")
axes[2].set_title("Job Satisfaction by Work Mode")

plt.tight_layout()
plt.show()
```

### 4. Business Output Explanation

The dashboard shows:

- **Salary hotspots** (high/low departments).

- **Attrition risk departments**.

- **Satisfaction differences** by work mode.

### 5. HR Strategy Use

- A one-stop view for **executive decision-making**.

- Helps in **quarterly HR reviews**.

- Enables **data-driven workforce strategy**.


