Case Study 6: Promotions and Performance Linkage

### 1. Business Objective

Leadership wants to evaluate whether **high performers are getting promoted fairly**. If promotions don’t align with performance, it may demotivate top talent.

### 2. Data Science Translation

- Create a **pivot table** with `PerformanceRating` vs. `PromotionsLast5Yrs`.

- Calculate **average promotions** for each rating level.

- Visualize as a heatmap-style table.

### 3. Python Code

```python
"""
Case Study 6: Promotions and Performance Linkage
------------------------------------------------
Demonstrates:
1. Pivot table creation
2. Group aggregation
3. Heatmap-style visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
hr_data = pd.read_excel(file_path, sheet_name="hr_dataset")

# Pivot table: Avg promotions by performance rating
promo_perf = hr_data.pivot_table(
    values="PromotionsLast5Yrs",
    index="PerformanceRating",
    aggfunc="mean"
)

# Visualization
plt.figure(figsize=(6,4))
plt.imshow(promo_perf, cmap="Blues", aspect="auto")
plt.colorbar(label="Avg Promotions (5 yrs)")
plt.xticks([])
plt.yticks(range(len(promo_perf.index)), promo_perf.index)
plt.title("Promotions vs Performance Ratings")
plt.xlabel("")
plt.ylabel("Performance Rating")
plt.show()
```

### 4. Business Output Explanation

- If high-rated employees have **similar or fewer promotions** than low-rated ones, it signals a **misalignment**.

- HR gets an **evidence-based view** of fairness in promotion practices.

### 5. HR Strategy Use

- Adjust **promotion criteria** to reward performance.

- Use this insight for **talent retention** planning.

- Avoid **“flight risk”** scenarios where top performers leave due to lack of recognition.
