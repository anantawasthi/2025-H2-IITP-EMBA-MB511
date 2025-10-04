Case Study 8: Job Satisfaction by Work Mode

### 1. Business Objective

Management wants to see if **remote, hybrid, or onsite employees** report different levels of **job satisfaction**.

### 2. Data Science Translation

- Aggregate average `JobSatisfaction` by `WorkMode`.

- Convert long → wide format for comparison.

- Plot results using a bar chart.

### 3. Python Code

```python
"""
Case Study 8: Job Satisfaction by Work Mode
-------------------------------------------
Demonstrates:
1. Long-to-wide transformation
2. Aggregation
3. Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Average satisfaction per work mode
satisfaction = hr_data.groupby('WorkMode')['JobSatisfaction'].mean().reset_index()

# Visualization
plt.figure(figsize=(7,5))
plt.bar(satisfaction['WorkMode'], satisfaction['JobSatisfaction'], color=["#8da0cb","#fc8d62","#66c2a5"])
plt.title("Average Job Satisfaction by Work Mode")
plt.xlabel("Work Mode")
plt.ylabel("Job Satisfaction (1-5)")
plt.show()
```

### 4. Business Output Explanation

- If remote workers show **higher satisfaction**, HR can highlight flexibility benefits.

- If onsite employees report **lower satisfaction**, interventions may be needed.

### 5. HR Strategy Use

- Shape **future workplace policies** (remote/hybrid balance).

- Improve **employee experience** across work modes.

- Justify investments in **flexible work infrastructure**.

---
