## Case Study 9: Attrition by Employment Type & Department

### 1. Business Objective

HR suspects **contract employees** leave more often in certain departments. They want to investigate attrition patterns by **EmploymentType & Department**.

### 2. Data Science Translation

- Group data by `EmploymentType` + `Department`.

- Calculate attrition % for each group.

- Export summary to Excel for reporting.

### 3. Python Code

```python
"""
Case Study 9: Attrition by Employment Type & Department
-------------------------------------------------------
Demonstrates:
1. GroupBy with multiple variables
2. Calculating percentages
3. Export for HR reports
"""

import pandas as pd

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Attrition rate by Employment Type & Department
attrition_summary = hr_data.groupby(['EmploymentType','Department'])['Attrition'].mean().reset_index()
attrition_summary['AttritionRatePct'] = attrition_summary['Attrition'] * 100

# Export results
attrition_summary.to_excel("Attrition_Report.xlsx", index=False)

attrition_summary.head()
```

### 4. Business Output Explanation

- The table highlights which **departments & employment types** suffer highest attrition.

- For example, contract workers in **Sales** may show 30% attrition vs only 8% for permanent staff.

### 5. HR Strategy Use

- Helps HR **target retention strategies** by department.

- Decide whether to **convert contractors to permanent roles**.

- Redesign **incentives** for high-attrition groups.

---

# 
