

# **Case Study 1: Understanding Salary Distribution Across Departments**

### 1. Business Objective

HR leadership wants to analyze whether **salary distribution differs significantly across departments**. This helps identify departments where compensation may lag, potentially causing dissatisfaction or attrition.

### 2. Data Science Translation

- Import HR dataset and clean missing salary values.

- Group employees by **Department**.

- Calculate **average monthly salary** per department.

- Visualize the salary distribution using a bar chart.

- Export the summarized table for HR reporting.

### 3. Python Code

```python
"""
Case Study 1: Salary Distribution Across Departments
----------------------------------------------------
This script demonstrates:
1. Data import & export
2. Handling missing values
3. Grouping and aggregation
4. Visualization with Matplotlib
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "/mnt/data/MB511_HR_Dataset.xlsx"
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Handle missing salary values by imputing with department mean
hr_data['MonthlySalaryINR'] = hr_data.groupby('Department')['MonthlySalaryINR']\
    .transform(lambda x: x.fillna(x.mean()))

# Aggregate average salary per department
dept_salary = hr_data.groupby('Department')['MonthlySalaryINR'].mean().reset_index()

# Export results for reporting
dept_salary.to_excel("Dept_Salary_Report.xlsx", index=False)

# Visualization
plt.figure(figsize=(10,6))
plt.bar(dept_salary['Department'], dept_salary['MonthlySalaryINR'], color="skyblue")
plt.xticks(rotation=45)
plt.title("Average Monthly Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Monthly Salary (INR)")
plt.tight_layout()
plt.show()
```

### 4. Business Output Explanation

The output is:

- An **Excel report** with average salary per department.

- A **bar chart** showing which departments pay higher or lower on average.

For example, HR might observe that **IT and Sales departments** pay higher average salaries, while **Operations and HR** have relatively lower compensation levels.

### 5. HR Strategy Use

This analysis helps HR in:

- Identifying departments with **potential pay gaps** that may impact retention.

- Making data-driven decisions about **budget allocation** for salary hikes.

- Building **compensation benchmarking reports** for leadership discussions.

- Ensuring **fairness and competitiveness** in pay across functions.


