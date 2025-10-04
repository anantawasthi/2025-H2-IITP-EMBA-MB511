## Case Study 4: Trend Analysis of Hiring Over Years

### 1. Business Objective

Management wants to understand **hiring patterns over the years** to check whether recruitment is keeping pace with organizational growth.

### 2. Data Science Translation

- Work with `HireDate` field (date conversions).

- Extract **year** from hire dates.

- Count hires per year.

- Plot hiring trend as a **line chart**.

### 3. Python Code

```python
"""
Case Study 4: Trend Analysis of Hiring
--------------------------------------
Demonstrates:
1. Working with dates
2. Extracting year from datetime
3. Aggregation
4. Line chart visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Extract year from HireDate
hr_data['HireYear'] = hr_data['HireDate'].dt.year

# Count hires by year
hire_trend = hr_data.groupby('HireYear')['EmployeeID'].count().reset_index()

# Visualization
plt.figure(figsize=(10,5))
plt.plot(hire_trend['HireYear'], hire_trend['EmployeeID'], marker='o', color="green")
plt.title("Hiring Trend Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Employees Hired")
plt.grid(True)
plt.show()
```

### 4. Business Output Explanation

The line chart shows **peaks and dips in hiring** — e.g., a spike in 2021 or a dip during recession years.

### 5. HR Strategy Use

- Helps HR decide whether hiring was **consistent with business growth**.

- Provides inputs for **future workforce planning**.

- Identifies **years with under-hiring**, which might have created skill gaps.
