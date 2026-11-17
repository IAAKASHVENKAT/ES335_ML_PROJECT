"""Weekly store sales forecasting. Run: python train.py Walmart.csv
Writes results.json, which the website (index.html) reads."""
import sys, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

SEED = 42
path = sys.argv[1] if len(sys.argv) > 1 else "Walmart.csv"

# ---- load + features -------------------------------------------------
df = pd.read_csv(path)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
raw = df.copy()  # untouched copy for the data summary
# last week's sales of the same store: known when we predict this week
df["lag1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
df["week"] = df["Date"].dt.isocalendar().week.astype(int)
df = df.dropna(subset=["lag1"]).reset_index(drop=True)

# ---- time-based split, fixed BEFORE any model is compared -------------
weeks = np.sort(df["Date"].unique())
c1 = pd.Timestamp(weeks[int(len(weeks) * 0.70)])
c2 = pd.Timestamp(weeks[int(len(weeks) * 0.85)])
train = df[df.Date < c1]
val = df[(df.Date >= c1) & (df.Date < c2)]
test = df[df.Date >= c2]
trval = pd.concat([train, val])

NUM = ["Temperature", "Fuel_Price", "CPI", "Unemployment", "lag1", "week"]
FEATS = NUM + ["Holiday_Flag", "Store"]
TARGET = "Weekly_Sales"


def metrics(y, p):
    e = np.asarray(y) - np.asarray(p)
    return {"MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "WMAPE": float(np.abs(e).sum() / np.abs(y).sum() * 100)}


def build(cls, params, num=None):
    num = NUM if num is None else num
    # scaler / encoder are inside the pipeline, so they are fit on training rows only
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("store", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Store"]),
        ("hol", "passthrough", ["Holiday_Flag"])])
    return Pipeline([("pre", pre), ("m", cls(**params))])


# ---- baselines (no learning, or almost none) ---------------------------
store_mean_val = val["Store"].map(train.groupby("Store")[TARGET].mean())
store_mean_test = test["Store"].map(trval.groupby("Store")[TARGET].mean())
val_rows = {"Last week's sales (naive)": metrics(val[TARGET], val["lag1"]),
            "Store average": metrics(val[TARGET], store_mean_val)}
test_rows = {"Last week's sales (naive)": metrics(test[TARGET], test["lag1"]),
             "Store average": metrics(test[TARGET], store_mean_test)}

# ---- models: tune on validation only -----------------------------------
space = {
    "Ridge": (Ridge, {"alpha": [0.1, 1, 10, 100]}),
    "Random forest": (RandomForestRegressor, {"n_estimators": [200], "min_samples_leaf": [1, 5, 10],
                                             "random_state": [SEED], "n_jobs": [-1]}),
    "Gradient boosting": (HistGradientBoostingRegressor, {"learning_rate": [0.03, 0.1],
                                                          "max_depth": [3, None], "random_state": [SEED]}),
}
best_params, final, full = {}, {}, {}
for name, (cls, grid) in space.items():
    scored = []
    for params in ParameterGrid(grid):
        m = build(cls, params).fit(train[FEATS], train[TARGET])
        scored.append((metrics(val[TARGET], m.predict(val[FEATS]))["MAE"], params))
    mae, params = min(scored, key=lambda t: t[0])
    full[name] = params
    m = build(cls, params).fit(train[FEATS], train[TARGET])
    val_rows[name] = metrics(val[TARGET], m.predict(val[FEATS]))
    best_params[name] = {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")}
    # test set is touched once per model, after tuning is finished
    final[name] = build(cls, params).fit(trval[FEATS], trval[TARGET])
    test_rows[name] = metrics(test[TARGET], final[name].predict(test[FEATS]))

best = min(space, key=lambda n: val_rows[n]["MAE"])  # chosen on validation, not test
pred = final[best].predict(test[FEATS])

# ---- error analysis ----------------------------------------------------
t = test.assign(pred=pred, naive=test["lag1"].values)
t["abs_err"] = (t[TARGET] - t["pred"]).abs()
hol = {}
for flag, g in t.groupby("Holiday_Flag"):
    hol["Holiday week" if flag == 1 else "Normal week"] = {
        "n": int(len(g)), "model_MAE": float(g["abs_err"].mean()),
        "naive_MAE": float((g[TARGET] - g["naive"]).abs().mean())}
worst = t.nlargest(5, "abs_err")[["Store", "Date", TARGET, "pred", "Holiday_Flag"]]
worst = [{"store": int(r.Store), "date": str(r.Date.date()), "actual": float(r.Weekly_Sales),
          "pred": float(r.pred), "holiday": int(r.Holiday_Flag)} for r in worst.itertuples()]

# ---- small-change check (post-hoc, NOT used to choose anything): drop feature groups
abl = {"Full model": test_rows[best]}
for label, drop in [("Without last week's sales (lag1)", ["lag1"]),
                    ("Without macro features (temp, fuel, CPI, unemployment)",
                     ["Temperature", "Fuel_Price", "CPI", "Unemployment"])]:
    num2 = [c for c in NUM if c not in drop]
    m2 = build(space[best][0], full[best], num2).fit(trval[num2 + ["Holiday_Flag", "Store"]], trval[TARGET])
    abl[label] = metrics(test[TARGET], m2.predict(test[num2 + ["Holiday_Flag", "Store"]]))

# ---- uncertainty: bootstrap over WEEKS (rows in the same week are not independent)
rng = np.random.default_rng(SEED)
g = t.assign(d=(t[TARGET] - t["naive"]).abs() - t["abs_err"]).groupby("Date")["d"].agg(["sum", "count"])
sm, ct = g["sum"].values, g["count"].values
boot = []
for _ in range(2000):
    i = rng.integers(0, len(sm), len(sm))
    boot.append(sm[i].sum() / ct[i].sum())
ci = {"gain_mean": float(sm.sum() / ct.sum()), "low": float(np.percentile(boot, 2.5)),
      "high": float(np.percentile(boot, 97.5)), "n_weeks": int(len(sm)), "n_boot": 2000}

# ---- permutation importance on validation (model fit on train only)
from sklearn.inspection import permutation_importance
mv = build(space[best][0], full[best]).fit(train[FEATS], train[TARGET])
pi = permutation_importance(mv, val[FEATS], val[TARGET], scoring="neg_mean_absolute_error",
                            n_repeats=5, random_state=SEED)
imp = sorted([{"feature": f, "mae_increase": float(m)} for f, m in zip(FEATS, pi.importances_mean)],
             key=lambda r: -r["mae_increase"])

# ---- data summary for the website
y = raw[TARGET]
sm_store = raw.groupby("Store")[TARGET].transform("mean")
ratio = y / sm_store
z = (y - sm_store) / raw.groupby("Store")[TARGET].transform("std")
tot = raw.groupby("Date")[TARGET].sum()
summary = {
    "n_rows": int(len(raw)), "missing": int(raw.isna().sum().sum()),
    "duplicates": int(raw.duplicated(["Store", "Date"]).sum()),
    "holiday_rows": int(raw.Holiday_Flag.sum()),
    "min": float(y.min()), "median": float(y.median()), "max": float(y.max()), "skew": float(y.skew()),
    "store_cv_median": float((raw.groupby("Store")[TARGET].std() / raw.groupby("Store")[TARGET].mean()).median()),
    "outliers_z3": int((z.abs() > 3).sum()),
    "corr_lag1": float(df[["lag1", TARGET]].corr().iloc[0, 1]),
    "ratio_holiday": float(ratio[raw.Holiday_Flag == 1].mean()), "ratio_normal": float(ratio[raw.Holiday_Flag == 0].mean()),
    "corr_macro": {c: float(np.corrcoef(raw[c], ratio)[0, 1]) for c in ["Temperature", "Fuel_Price", "CPI", "Unemployment"]},
    "store_means": [[int(k), float(v)] for k, v in raw.groupby("Store")[TARGET].mean().items()],
    "total_series": [[str(k.date()), float(v)] for k, v in tot.items()],
}

out = {
    "dataset": {"rows_used": int(len(df)), "stores": int(df.Store.nunique()),
                "first_date": str(df.Date.min().date()), "last_date": str(df.Date.max().date())},
    "split": {"train": [str(train.Date.min().date()), str(train.Date.max().date()), int(len(train))],
              "val": [str(val.Date.min().date()), str(val.Date.max().date()), int(len(val))],
              "test": [str(test.Date.min().date()), str(test.Date.max().date()), int(len(test))]},
    "best_model": best, "ablation": abl, "ci": ci, "importance": imp, "summary": summary, "best_params": best_params,
    "validation": val_rows, "test": test_rows, "holiday": hol, "worst": worst,
    "preds": [{"s": int(r.Store), "d": str(r.Date.date()), "y": float(r.Weekly_Sales),
               "p": float(r.pred), "n": float(r.naive), "h": int(r.Holiday_Flag)} for r in t.itertuples()],
}
json.dump(out, open("results.json", "w"))
print("best (by validation MAE):", best)
print(pd.DataFrame(test_rows).T.round(2))
