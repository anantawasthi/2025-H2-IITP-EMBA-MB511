## Introduction to Data Management and Visualization

### Pandas and Matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load once
df = pd.read_excel("/mnt/data/MB511_HR_Dataset.xlsx", sheet_name="hr_dataset")
```

---

**Dataset Exploration**

```python
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load the dataset
# -----------------------------
file_path = "/mnt/data/MB511_HR_Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="hr_dataset")

# -----------------------------
# 1. Basic structure
# -----------------------------
print("Shape of dataset:", df.shape)            # rows, columns
print("Column names:\n", df.columns.tolist())   # all fields

# -----------------------------
# 2. Data types & nulls
# -----------------------------
print("\nData types and non-null counts:")
print(df.info())

# Quick missing values summary
print("\nMissing values per column:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# -----------------------------
# 3. First few rows (sanity check)
# -----------------------------
print("\nSample records:")
print(df.head())

# -----------------------------
# 4. Basic descriptive stats
# -----------------------------
print("\nSummary statistics (numeric):")
print(df.describe().T[['mean','std','min','max']])

# -----------------------------
# 5. Quick categorical profile
# -----------------------------
cat_cols = ['Gender','MaritalStatus','Department','JobRole',
            'EmploymentType','WorkMode','EducationLevel']
for col in cat_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head())

# -----------------------------
# 6. Simple visualization previews
# -----------------------------
# Gender distribution
df['Gender'].value_counts().plot(kind='bar', color=['#66b3ff','#ff9999'])
plt.title("Gender Distribution"); plt.xlabel("Gender"); plt.ylabel("Count")
plt.show()

# Attrition ratio
df['Attrition'].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%')
plt.title("Attrition Share (Active vs Left)")
plt.ylabel("")
plt.show()
```

---

### 1) Import from Excel and inspect shape

```python
# Basic load + quick sanity check
df = pd.read_excel("/mnt/data/MB511_HR_Dataset.xlsx", sheet_name="hr_dataset")
print(df.shape); print(df.columns.tolist())
```

### 2) Export a trimmed reporting view

```python
# Keep just a few business columns for a quick share
cols = ["EmployeeID","Department","JobRole","MonthlySalaryINR","PerformanceRating","Attrition"]
df[cols].to_excel("reporting_view.xlsx", index=False)
```

### 3) Convert text-like cols to category for memory + consistency

```python
# Speeds up groupby/filters on high-cardinality text
cat_cols = ["Gender","MaritalStatus","Department","JobRole","EmploymentType","WorkMode","EducationLevel"]
df[cat_cols] = df[cat_cols].astype("category")
```

### 4) Coerce salary to numeric (fix stray text)

```python
# Non-numeric junk becomes NaN; keeps pipeline robust
df["MonthlySalaryINR"] = pd.to_numeric(df["MonthlySalaryINR"], errors="coerce")
```

### 5) Create a readable performance label (numeric → string)

```python
# Good for charts/legends without numeric ticks
df["PerfLabel"] = "P" + df["PerformanceRating"].astype(str)
```

### 6) Work with dates: extract year/month/day from HireDate

```python
# Enables trends, cohorts, seasonality analyses
df["HireYear"]  = df["HireDate"].dt.year
df["HireMonth"] = df["HireDate"].dt.month
df["HireDay"]   = df["HireDate"].dt.day
```

### 7) Derive Annual CTC (Monthly×12 + Annual Bonus)

```python
# Simple comp metric HR leaders expect in dashboards
df["AnnualCTC"] = df["MonthlySalaryINR"]*12 + df["AnnualBonusINR"]
```

### 8) Business rule flag: “At-Risk” (low pay & low satisfaction)

```python
# Median-based—simple, explainable thresholding
med_sal = df["MonthlySalaryINR"].median()
df["AtRiskFlag"] = (df["MonthlySalaryINR"] < med_sal) & (df["JobSatisfaction"] <= 2)
```

### 9) Sort by salary (desc) for compensation review

```python
# Quick who’s-who for comp discussions
df_sorted = df.sort_values("MonthlySalaryINR", ascending=False)
```

### 10) Multi-key sort: Department ↑ then Salary ↓

```python
# Useful for within-department pay comparisons
df_sorted2 = df.sort_values(["Department","MonthlySalaryINR"], ascending=[True, False])
```

### 11) Rename a couple of columns for external reporting

```python
# Cleaner labels for exec decks
df = df.rename(columns={"MonthlySalaryINR":"MonthlySalary","PromotionsLast5Yrs":"Promotions5Y"})
```

### 12) Filter current employees (Attrition==0)

```python
# Common slice for engagement/retention work
active = df[df["Attrition"] == 0]
```

### 13) Filter: Top performers in Sales (rating ≥4)

```python
# Drill into high-impact talent pockets
sales_top = df[(df["Department"]=="Sales") & (df["PerformanceRating"]>=4)]
```

### 14) GroupBy: average salary and attrition by Department

```python
# Two core HR KPIs in one table
dept_kpis = (df.groupby("Department", as_index=False)
               .agg(AvgSalary=("MonthlySalary","mean"),
                    AttritionRate=("Attrition","mean")))
dept_kpis["AttritionPct"] = (dept_kpis["AttritionRate"]*100).round(2)
```

### 15) Pivot: average Job Satisfaction by Department × WorkMode

```python
# Matrix view for policy debates (remote/hybrid/onsite)
sat_pivot = df.pivot_table(values="JobSatisfaction",
                           index="Department", columns="WorkMode",
                           aggfunc="mean")
```

### 16) Melt (wide→long): reshape pay components for plotting

```python
# Long format plays nicely with many visuals
pay_long = pd.melt(df,
                   id_vars=["Department"],
                   value_vars=["MonthlySalary","AnnualBonusINR"],
                   var_name="PayType", value_name="Amount")
```

### 17) Pivot (long→wide): headcount by EmploymentType per Department

```python
# Instant org structure snapshot
headcount = df.pivot_table(index="Department", columns="EmploymentType",
                           values="EmployeeID", aggfunc="count", fill_value=0)
```

### 18) Merge with a small department target table

```python
# Classic join for variance-to-target analysis
targets = pd.DataFrame({"Department": df["Department"].unique(),
                        "HeadcountTarget": 100})
merged = df.merge(targets, on="Department", how="left")
```

### 19) Missing values audit (descending)

```python
# Fast way to spot data quality hotspots
na_report = df.isna().sum().sort_values(ascending=False)
print(na_report.head(10))
```

### 20) Impute MonthlySalary within Department (mean)

```python
# Context-aware, keeps distributions reasonable
df["MonthlySalary"] = (df.groupby("Department")["MonthlySalary"]
                         .transform(lambda s: s.fillna(s.mean())))
```

### 21) Outliers in LeaveDaysAnnual via IQR

```python
# Transparent rule HR can explain to leadership
Q1, Q3 = df["LeaveDaysAnnual"].quantile([0.25, 0.75])
IQR = Q3 - Q1
mask = (df["LeaveDaysAnnual"] < Q1 - 1.5*IQR) | (df["LeaveDaysAnnual"] > Q3 + 1.5*IQR)
leave_outliers = df[mask]
```

### 22) Cap (winsorize) extreme LeaveDaysAnnual at IQR bounds

```python
# Gentle treatment without dropping rows
low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df["LeaveDaysAnnual_Capped"] = df["LeaveDaysAnnual"].clip(lower=low, upper=high)
```

### 23) Quick bar: average salary by Department

```python
# Minimalist visual for comp committee
(df.groupby("Department")["MonthlySalary"].mean()
   .plot(kind="bar"))
plt.title("Average Monthly Salary by Department"); plt.ylabel("INR")
plt.tight_layout(); plt.show()
```

### 24) Line: hiring trend by year

```python
# Detect ramp-ups / freezes quickly
(df["HireDate"].dt.year.value_counts()
   .sort_index()
   .plot(kind="line", marker="o"))
plt.title("Hiring Trend by Year"); plt.xlabel("Year"); plt.ylabel("Hires")
plt.grid(True); plt.show()
```

### 25) Scatter: TrainingHours vs PerformanceRating

```python
# Visual read on ROI of training
plt.scatter(df["TrainingHours"], df["PerformanceRating"], alpha=0.5)
plt.xlabel("Training Hours"); plt.ylabel("Performance Rating")
plt.title("Training vs Performance"); plt.show()
```


