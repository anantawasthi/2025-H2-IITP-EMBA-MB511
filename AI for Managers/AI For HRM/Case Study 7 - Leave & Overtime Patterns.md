Case Study 7: Leave & Overtime Patterns

### 1. Business Objective

HR suspects that **excessive overtime** is linked to **higher absenteeism**. They want to validate if this trend exists.

### 2. Data Science Translation

- Filter employees with **high overtime (>30 hrs/month)**.

- Compare their **annual leave days** with others.

- Detect **outliers** in leave data.

### 3. Python Code

```python
"""
Case Study 7: Leave & Overtime Patterns
---------------------------------------
Demonstrates:
1. Filtering with multiple conditions
2. Outlier detection
3. Comparison analysis
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# High overtime group
high_ot = hr_data[hr_data['OvertimeHoursMonthly'] > 30]

# Compare average leave days
avg_leave_high = high_ot['LeaveDaysAnnual'].mean()
avg_leave_all = hr_data['LeaveDaysAnnual'].mean()

print(f"Avg Leave Days (High OT): {avg_leave_high:.2f}")
print(f"Avg Leave Days (All Employees): {avg_leave_all:.2f}")

# Outlier detection using IQR
Q1 = hr_data['LeaveDaysAnnual'].quantile(0.25)
Q3 = hr_data['LeaveDaysAnnual'].quantile(0.75)
IQR = Q3 - Q1
outliers = hr_data[(hr_data['LeaveDaysAnnual'] < Q1 - 1.5*IQR) | (hr_data['LeaveDaysAnnual'] > Q3 + 1.5*IQR)]
```

### 4. Business Output Explanation

- If high-overtime employees take **significantly more leave**, it suggests burnout.

- Outlier detection highlights employees with **extreme absenteeism**.

### 5. HR Strategy Use

- Implement **work-life balance policies**.

- Identify **burnout risks** early.

- Redesign overtime/shift policies.
