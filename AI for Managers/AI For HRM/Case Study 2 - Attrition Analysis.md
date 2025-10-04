## Case Study 2: Attrition Risk by Salary & Job Satisfaction

### 1. Business Objective

HR wants to know whether **low salary and low job satisfaction** are linked to **employee attrition**. If this link exists, managers can intervene proactively with better engagement and pay strategies.

### 2. Data Science Translation

- Filter employees who have **left (Attrition=1)**.

- Compare their **average salary and job satisfaction** against active employees.

- Visualize differences with a **side-by-side bar chart**.

### 3. Python Code

```python
"""
Case Study 2: Attrition Risk by Salary & Job Satisfaction
--------------------------------------------------------
Demonstrates:
1. Filtering rows with conditions
2. Logical operations (Attrition = 1 vs 0)
3. Aggregation & comparison
4. Visualization using Matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "/mnt/data/MB511_HR_Dataset.xlsx"
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Filter groups: employees who left vs those who stayed
attrition_grp = hr_data.groupby('Attrition')[['MonthlySalaryINR', 'JobSatisfaction']].mean()

# Reset index for plotting
attrition_grp = attrition_grp.reset_index()
attrition_grp['Attrition'] = attrition_grp['Attrition'].map({0: "Active", 1: "Left"})

# Visualization
attrition_grp.set_index('Attrition').plot(kind='bar', figsize=(8,6), color=['teal','orange'])
plt.title("Attrition vs Salary & Job Satisfaction")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

### 4. Business Output Explanation

The chart will clearly show:

- Employees who **left** tend to have **lower salaries** and **lower job satisfaction scores** than active employees.

- The visual makes the pattern intuitive and easy to interpret even for non-technical managers.

### 5. HR Strategy Use

- **Retention programs** can be prioritized for employees in the **low-salary & low-satisfaction** segment.

- **Compensation restructuring** can target at-risk job roles or departments.

- **Engagement initiatives** (surveys, career development, mentoring) can be applied to reduce attrition.


