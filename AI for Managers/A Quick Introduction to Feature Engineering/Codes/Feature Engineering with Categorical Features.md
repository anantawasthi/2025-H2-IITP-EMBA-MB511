## 1) Label Encoding (good for **ordinal/binary** categories)

```python
"""
LABEL ENCODING
Maps categories to integers: e.g., {"Low":0, "Medium":1, "High":2}.
Use for ordered categories (ordinal). For purely nominal categories (like City), prefer one-hot.

What we do:
- Fit LabelEncoder on the training set only.
- Transform both train and test sets.
"""
from sklearn.preprocessing import LabelEncoder

train_le = train_df.copy()
test_le = test_df.copy()

le = LabelEncoder()
# Ordinal example: Satisfaction
le.fit(train_le["Satisfaction"])

train_le["Satisfaction_LE"] = le.transform(train_le["Satisfaction"])
test_le["Satisfaction_LE"] = le.transform(test_le["Satisfaction"])

print(train_le[["Satisfaction","Satisfaction_LE"]].head(5))
print(test_le[["Satisfaction","Satisfaction_LE"]].head(5))
```

---

### 2) One-Hot Encoding (good for **low-cardinality** nominal categories)

```python
"""
ONE-HOT ENCODING
Creates binary columns for each category (e.g., City -> City_Mumbai, City_Delhi, ...).
Use when there are relatively few categories; avoids implying any order.

What we do:
- Use pandas.get_dummies on the combined columns list derived from train.
- Align columns in test to match train (filling missing columns with 0).
"""
# Choose a nominal variable
onehot_cols = ["City", "Gender"]

# Fit on train: find all categories present
train_onehot = pd.get_dummies(train_df[onehot_cols], drop_first=False)

# Transform test using the same categories (align columns)
test_onehot = pd.get_dummies(test_df[onehot_cols], drop_first=False)
test_onehot = test_onehot.reindex(columns=train_onehot.columns, fill_value=0)

print(train_onehot.head(3))
print(test_onehot.head(3))
```

---

### 3) Ordinal Encoding (requires a **defined order**)

```python
"""
ORDINAL ENCODING
Encodes categories with a meaningful order (e.g., Low < Medium < High).

What we do:
- Define the order explicitly.
- Map to integers accordingly.
"""
# Define order for Satisfaction
order = {"Low": 0, "Medium": 1, "High": 2}

train_ord = train_df.copy()
test_ord = test_df.copy()

train_ord["Satisfaction_OE"] = train_ord["Satisfaction"].map(order)
test_ord["Satisfaction_OE"] = test_ord["Satisfaction"].map(order)

print(train_ord[["Satisfaction","Satisfaction_OE"]].head(5))
print(test_ord[["Satisfaction","Satisfaction_OE"]].head(5))
```

---

### 4) Frequency Encoding (replace category with its **relative frequency**)

```python
"""
FREQUENCY ENCODING
Replaces each category with how often it appears in the training data.

Why:
- Captures how "common" or "rare" a category is.
- Useful for large datasets with many categories.

What we do:
- Compute frequencies on train only.
- Map both train and test; unknown test categories get NaN (fill with 0).
"""
train_freq = train_df.copy()
test_freq = test_df.copy()

# Compute normalized frequencies on train
city_freq = train_freq["City"].value_counts(normalize=True)

train_freq["City_FreqEnc"] = train_freq["City"].map(city_freq)
test_freq["City_FreqEnc"] = test_freq["City"].map(city_freq).fillna(0.0)

print(train_freq[["City","City_FreqEnc"]].head(5))
print(test_freq[["City","City_FreqEnc"]].head(5))
```

---

### 5) Target (Mean) Encoding with **K-Fold** (reduces leakage/overfitting)

```python
"""
TARGET (MEAN) ENCODING — LEAKAGE-SAFE WITH K-FOLD
Replaces each category with the average target (e.g., churn rate) for that category.
We do this using K-fold on the training data to avoid leaking target information.

Steps:
1) On train: For each fold, compute category->mean using ONLY the other folds.
2) Transform the held-out fold with those means.
3) For test: use overall category means computed on full train (no target from test).
"""
from sklearn.model_selection import KFold

def kfold_target_encode(train, test, cat_col, target_col, n_splits=5, smoothing=0.0):
    """
    K-fold target encoding for a single categorical column.
    smoothing: optional blending toward global mean (0.0 = no smoothing).
    """
    train_encoded = train.copy()
    test_encoded = test.copy()

    global_mean = train[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Placeholder for out-of-fold encodings
    oof = pd.Series(index=train.index, dtype=float)

    for tr_idx, val_idx in kf.split(train):
        tr_fold, val_fold = train.iloc[tr_idx], train.iloc[val_idx]
        # Calculate means on training fold only
        means = tr_fold.groupby(cat_col)[target_col].mean()

        if smoothing > 0:
            # Optional simple smoothing by counts
            counts = tr_fold.groupby(cat_col)[target_col].count()
            means = (means * counts + global_mean * smoothing) / (counts + smoothing)

        # Map into validation fold
        oof.iloc[val_idx] = val_fold[cat_col].map(means)

    # Fill unseen categories with global mean
    oof = oof.fillna(global_mean)
    train_encoded[f"{cat_col}_TE"] = oof

    # For test: compute means using full train
    full_means = train.groupby(cat_col)[target_col].mean()
    if smoothing > 0:
        counts = train.groupby(cat_col)[target_col].count()
        full_means = (full_means * counts + global_mean * smoothing) / (counts + smoothing)

    test_encoded[f"{cat_col}_TE"] = test[cat_col].map(full_means).fillna(global_mean)
    return train_encoded, test_encoded

# Apply to 'City' using churn as target
train_te, test_te = kfold_target_encode(train_df, test_df, cat_col="City", target_col="Churn", n_splits=5, smoothing=10.0)

print(train_te[["City","Churn","City_TE"]].head(5))
print(test_te[["City","Churn","City_TE"]].head(5))
```

---

### 6) Binary / Hash Encoding (efficient for **high cardinality**)

```python
"""
BINARY / HASH ENCODING
Efficiently encodes high-cardinality categorical features (e.g., Product_ID with 1000+ values)
into a compact set of numeric columns.

Tools:
- category_encoders.BinaryEncoder
- category_encoders.HashingEncoder (stateless; set n_components to control width)

Note: These methods are less interpretable than one-hot/ordinal encoders.
"""
# If not installed: pip install category_encoders
import category_encoders as ce

# Binary encoding
train_bin = train_df.copy()
test_bin = test_df.copy()

bin_encoder = ce.BinaryEncoder(cols=["Product_ID"], return_df=True)
train_bin = bin_encoder.fit_transform(train_bin)
test_bin = bin_encoder.transform(test_bin)

print(train_bin.filter(like="Product_ID_").head(3))
print(test_bin.filter(like="Product_ID_").head(3))

# Hashing encoding (example with 8 hashed components)
train_hash = train_df.copy()
test_hash = test_df.copy()

hash_encoder = ce.HashingEncoder(cols=["Product_ID"], n_components=8, return_df=True)
train_hash = hash_encoder.fit_transform(train_hash)
test_hash = hash_encoder.transform(test_hash)

print(train_hash.filter(like="Product_ID_").head(3))
print(test_hash.filter(like="Product_ID_").head(3))
```

---

### 7) Rare-Category Grouping (collapsing very infrequent levels into **“Other”**)

```python
"""
RARE-CATEGORY GROUPING
Groups categories that appear very infrequently to reduce sparsity and stabilize model training.

What we do:
- Compute category counts on the training set.
- Replace categories below a minimum frequency threshold with 'Other' in both train and test.
"""
train_rare = train_df.copy()
test_rare = test_df.copy()

min_count = int(0.01 * len(train_rare))  # e.g., 1% threshold
counts = train_rare["City"].value_counts()
rare_levels = counts[counts < min_count].index

train_rare["City_Grouped"] = train_rare["City"].where(~train_rare["City"].isin(rare_levels), "Other")
test_rare["City_Grouped"] = test_rare["City"].where(~test_rare["City"].isin(rare_levels), "Other")

print(train_rare[["City","City_Grouped"]].head(10))
print(test_rare[["City","City_Grouped"]].head(10))
```

---

### 8) Putting it together (simple pipeline idea)

```python
"""
PIPELINE IDEA (conceptual sketch)
A practical project typically combines multiple steps:
- Rare-category grouping
- One-hot or target encoding (depending on cardinality)
- Scaling numeric features (if needed)
- Model training with cross-validation

Below is a conceptual outline without fitting a final model.
"""

# Example: rare grouping on City -> one-hot
train_pipe = train_df.copy()
test_pipe = test_df.copy()

min_count = int(0.01 * len(train_pipe))
rare_levels = train_pipe["City"].value_counts()
rare_levels = rare_levels[rare_levels < min_count].index

train_pipe["City_Grouped"] = train_pipe["City"].where(~train_pipe["City"].isin(rare_levels), "Other")
test_pipe["City_Grouped"] = test_pipe["City"].where(~test_pipe["City"].isin(rare_levels), "Other")

# One-hot encode the grouped city and gender
train_X = pd.get_dummies(train_pipe[["City_Grouped","Gender"]], drop_first=False)
test_X  = pd.get_dummies(test_pipe[["City_Grouped","Gender"]], drop_first=False)

# Align columns
test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

# Add a couple of numeric features (already in numeric form)
train_X = pd.concat([train_X, train_pipe[["Age","Monthly_Income","Total_Spend"]].reset_index(drop=True)], axis=1)
test_X  = pd.concat([test_X,  test_pipe[["Age","Monthly_Income","Total_Spend"]].reset_index(drop=True)], axis=1)

train_y = train_pipe["Churn"]
test_y  = test_pipe["Churn"]

print("Training matrix shape:", train_X.shape)
print("Test matrix shape:", test_X.shape)
```

---

If you want, I can now:

- wrap one or two of these encoders into a **sklearn `ColumnTransformer`/`Pipeline`**, or

- proceed to **date/time feature engineering** with the same structure (intro table + Python code).
