## Method A — Date-Part Extraction (year, month, weekday, quarter)

**Introduction**  
Breaks a timestamp into calendar parts (year, month, weekday, quarter) so models can learn easy, interpretable periodic patterns such as seasonality and working-day effects. These features are human-readable, quick to validate, and often provide strong baselines before more advanced temporal encodings.

**When to use**   
Use when behavior varies by calendar units—weekday traffic, month/quarter sales cycles, fiscal periods. Works well for tree models and reporting dashboards.

**When not to use**  
Avoid depending solely on parts when wrap-around proximity matters (Dec ~ Jan) or when trends dominate (year-on-year drift). Consider cyclical encodings or detrending.

```python
"""
DATE-PART EXTRACTION
Adds: LP_Year, LP_Month, LP_Weekday, LP_Quarter from Last_Purchase_Date.
"""
df["Last_Purchase_Date"] = pd.to_datetime(df["Last_Purchase_Date"])
df["LP_Year"]    = df["Last_Purchase_Date"].dt.year
df["LP_Month"]   = df["Last_Purchase_Date"].dt.month        # 1..12
df["LP_Weekday"] = df["Last_Purchase_Date"].dt.weekday      # 0=Mon..6=Sun
df["LP_Quarter"] = df["Last_Purchase_Date"].dt.quarter      # 1..4

df[["Last_Purchase_Date","LP_Year","LP_Month","LP_Weekday","LP_Quarter"]].head()
```

---

## Method B — Cyclical Encoding (sin/cos)

**Introduction**  
Maps periodic units (month, weekday, hour) to sine/cosine on a circle so adjacent values remain close and boundary jumps (12→1, 6→0) vanish. Improves linear, distance-based, and neural models that otherwise misinterpret cyclical proximity.

**When to use**  
Use for truly periodic features (weekday, month, hour), especially with linear/NN/KNN models.

**When not to use**  
Don’t use for non-periodic values (e.g., year). For tree models, benefits may be marginal because trees split discretely.

```python
"""
CYCLICAL ENCODING
Creates sin/cos pairs for month (12) and weekday (7).
"""
import numpy as np

if "LP_Month" not in df:  # safety if A not run
    df["LP_Month"]   = df["Last_Purchase_Date"].dt.month
if "LP_Weekday" not in df:
    df["LP_Weekday"] = df["Last_Purchase_Date"].dt.weekday

df["LP_Month_sin"]   = np.sin(2*np.pi * df["LP_Month"] / 12.0)
df["LP_Month_cos"]   = np.cos(2*np.pi * df["LP_Month"] / 12.0)
df["LP_Weekday_sin"] = np.sin(2*np.pi * df["LP_Weekday"] / 7.0)
df["LP_Weekday_cos"] = np.cos(2*np.pi * df["LP_Weekday"] / 7.0)

df[["LP_Month","LP_Month_sin","LP_Month_cos","LP_Weekday","LP_Weekday_sin","LP_Weekday_cos"]].head()
```

---

## Method C — Recency & Tenure (“time-since” features)

**Introduction**  
Quantifies elapsed time since a key event (recency) and since onboarding (tenure). Recency is a powerful signal for churn/propensity; tenure approximates lifecycle stage, often correlating with loyalty and value.

**When to use**  
Use in retention, marketing, risk, and engagement where time gaps matter.

**When not to use**  
Avoid when timestamps are unreliable or events are non-recurring and not temporally meaningful.

```python
"""
RECENCY & TENURE
Adds: Days_Since_Last_Purchase, Tenure_Days, and an interpretable Recency_Bucket.
"""
df["AsOfDate"] = pd.to_datetime(df["AsOfDate"])
df["Days_Since_Last_Purchase"] = (df["AsOfDate"] - df["Last_Purchase_Date"]).dt.days
df["Tenure_Days"]              = (df["AsOfDate"] - df["Signup_Date"]).dt.days

bins   = [0, 30, 90, 180, 365, 730, 10000]
labels = ["≤30d","31–90d","91–180d","181–365d","366–730d",">730d"]
df["Recency_Bucket"] = pd.cut(df["Days_Since_Last_Purchase"], bins=bins, labels=labels, include_lowest=True)

# sanity
assert (df["Days_Since_Last_Purchase"] >= 0).all()
assert (df["Tenure_Days"] >= 0).all()

df[["Days_Since_Last_Purchase","Recency_Bucket","Tenure_Days"]].head()
```

---

## Method D — Weekend & Holiday Indicators

**Introduction**  
Binary flags for weekends and holidays capture non-working day effects that drive demand spikes, staffing differences, and behavior shifts—simple, interpretable, and often effective.

**When to use**  
Use when operations/behavior change on weekends/holidays; ensure geography-specific calendars.

**When not to use**  
Avoid with mixed regions/time zones without proper local calendars; avoid if timestamps lack user locale context.

```python
"""
WEEKEND & HOLIDAY FLAGS
Adds: LP_IsWeekend and LP_IsHoliday (illustrative holiday list—customize per locale).
"""
df["LP_IsWeekend"] = df["Last_Purchase_Date"].dt.weekday.isin([5,6]).astype(int)

simple_holidays = pd.to_datetime([
    "2025-01-26",  # Republic Day (example)
    "2025-03-14",  # Holi (approx.)
    "2025-08-15",  # Independence Day
    "2025-10-20",  # Dussehra (approx.)
    "2025-10-29",  # Diwali (approx.)
])
df["LP_IsHoliday"] = df["Last_Purchase_Date"].dt.normalize().isin(simple_holidays).astype(int)

df[["Last_Purchase_Date","LP_IsWeekend","LP_IsHoliday"]].head()
```

---

## Method E — Seasonality / Festive Buckets & Quarter Dummies

**Introduction**  
Encodes known seasonal regimes (e.g., festive peaks) and quarter indicators to capture systematic demand shifts. Leverages domain knowledge to create interpretable, policy-relevant features.

**When to use**  
Use when domain knowledge identifies peak/off-peak seasons or quarter effects.

**When not to use**  
Avoid when history is too short/irregular to support stable seasonal inference.

```python
"""
SEASONALITY & QUARTERS
Adds: LP_IsFestivePeak (Oct/Nov demo) and one-hot quarter dummies.
"""
if "LP_Month" not in df:
    df["LP_Month"] = df["Last_Purchase_Date"].dt.month
if "LP_Quarter" not in df:
    df["LP_Quarter"] = df["Last_Purchase_Date"].dt.quarter

df["LP_IsFestivePeak"] = df["LP_Month"].isin([10,11]).astype(int)
df = pd.get_dummies(df, columns=["LP_Quarter"], prefix="LP_Q", drop_first=False)

df[["LP_Month","LP_IsFestivePeak","LP_Q_1","LP_Q_2","LP_Q_3","LP_Q_4"]].head()
```

---

## Method F — Rolling Window Aggregations (30/90-day)

**Introduction**  
Summarizes recent event-level behavior (counts, sums, averages) over fixed windows (e.g., 30/90 days). Converts raw event streams into compact, recency-weighted signals—great for churn, CLV, conversion, or risk models.

**When to use**  
Use with event-level tables (transactions, logins) where recent activity drives outcomes.

**When not to use**  
Avoid when you only have a static snapshot or when future events leak into training windows.

```python
"""
ROLLING 30/90-DAY AGGREGATES
Creates a synthetic transactions table and computes rolling aggregates per customer.
In production, replace the synthetic part with your actual events table.
"""
# --- fabricate transactions over last 12 months for a subset ---
rng = np.random.default_rng(11)
n_tx_customers = 2000
tx_customers = df["Customer_ID"].sample(n_tx_customers, random_state=42).tolist()

rows = []
start_12m = df["AsOfDate"].iloc[0] - pd.Timedelta(days=365)
for cid in tx_customers:
    k = int(rng.integers(5, 41))
    dates = pd.to_datetime(rng.integers(start_12m.value//10**9, df["AsOfDate"].iloc[0].value//10**9, size=k), unit="s")
    amounts = np.clip(rng.normal(2000, 800, k), 100, 20000)
    rows.extend([(cid, d, float(a)) for d, a in zip(dates, amounts)])

tx = pd.DataFrame(rows, columns=["Customer_ID","Tx_Date","Tx_Amount"]).sort_values(["Customer_ID","Tx_Date"])

# --- compute rolling features per customer (time-indexed) ---
t = tx.set_index("Tx_Date").sort_index()

def _roll(g: pd.DataFrame) -> pd.DataFrame:
    """Per-customer rolling windows: 30D/90D count, sum, avg."""
    g = g.sort_index()
    g["Tx_Count_30d"] = g["Tx_Amount"].rolling("30D").count()
    g["Tx_Sum_30d"]   = g["Tx_Amount"].rolling("30D").sum()
    g["Tx_Avg_30d"]   = g["Tx_Amount"].rolling("30D").mean()
    g["Tx_Count_90d"] = g["Tx_Amount"].rolling("90D").count()
    g["Tx_Sum_90d"]   = g["Tx_Amount"].rolling("90D").sum()
    g["Tx_Avg_90d"]   = g["Tx_Amount"].rolling("90D").mean()
    return g

tx_feats = (t.groupby("Customer_ID", group_keys=False)
              .apply(_roll)
              .reset_index())

# --- take latest snapshot per customer and join back to df ---
latest = (tx_feats.sort_values(["Customer_ID","Tx_Date"])
                 .groupby("Customer_ID").tail(1))

join_cols = ["Customer_ID","Tx_Count_30d","Tx_Sum_30d","Tx_Avg_30d","Tx_Count_90d","Tx_Sum_90d","Tx_Avg_90d"]
df = df.merge(latest[join_cols], on="Customer_ID", how="left")

# fill customers with no events
fill_cols = [c for c in join_cols if c != "Customer_ID"]
df[fill_cols] = df[fill_cols].fillna(0.0)

df.filter(regex="^Tx_").head()
```

---

## Method G — Lag Features (per-entity daily time series)

**Introduction**  
Shifts past values forward (e.g., t−1, t−7) so the model can learn autocorrelation and short memory effects. Common in forecasting and anomaly detection pipelines.

**When to use**  
Use when you have dense, regular time series per entity and the past informs the near future.

**When not to use**  
Avoid with sparse/irregular data or when regime shifts make recent lags misleading.

```python
"""
LAG FEATURES
Aggregates transactions to daily spend per customer and creates Lag_1 and Lag_7.
"""
daily = (tx.groupby(["Customer_ID", pd.Grouper(key="Tx_Date", freq="D")])["Tx_Amount"]
           .sum()
           .reset_index()
           .rename(columns={"Tx_Amount":"Daily_Spend"}))

daily = daily.sort_values(["Customer_ID","Tx_Date"])
daily["Lag_1"] = daily.groupby("Customer_ID")["Daily_Spend"].shift(1)
daily["Lag_7"] = daily.groupby("Customer_ID")["Daily_Spend"].shift(7)

daily.head(10)
```

---

### 
