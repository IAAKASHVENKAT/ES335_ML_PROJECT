# Predicting Retail Sales Using Machine Learning

ES 335 course project (undergraduate track)
Hamsika Para (23110230) and Aakash Venkatesan (23110002)

- Live website: <PASTE GITHUB PAGES URL, e.g. https://USERNAME.github.io/REPO/>
- Source repository: <PASTE REPO URL, e.g. https://github.com/USERNAME/REPO>

## Question
Can a store's sales for the coming week be forecast from information known at the start of that week, and does a machine-learning model beat simple rules? One row = one store in one week; target = `Weekly_Sales`.

## Repository contents and code location
| File | What it is |
|---|---|
| `index.html` | The project website (static HTML/CSS/JS, no build step). Reads `results.json`; a copy is embedded so it also works when opened as a local file. |
| `train.py` | All the code: loading, features, time-based split, baselines, tuning, final test, error analysis, bootstrap interval, permutation importance, data summary. Writes `results.json`. |
| `results.json` | Output of `train.py`. Every number and chart on the site comes from here. |
| `Walmart.csv` | The dataset (6,435 rows), included so the project runs as is. |
| `requirements.txt` | Python library versions used (Python 3.12). |

## Data access
- Source: the Walmart sales dataset on Kaggle (a free Kaggle account is needed to download it from there): <PASTE THE EXACT KAGGLE URL YOU DOWNLOADED FROM>
- Columns: Store, Date (dd-mm-yyyy), Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment.
- 6,435 rows = 45 stores x 143 weeks (2010-02-05 to 2012-10-26). No missing values, no duplicate store-weeks.
- The CSV is in this repository, so no download is needed to reproduce the results. Please check the Kaggle dataset's licence/terms before redistributing it elsewhere.

## Setup and rebuild
```
git clone <repository URL>
cd <repository folder>
pip install -r requirements.txt
python train.py Walmart.csv          # about a minute; writes results.json (seed 42)
python -m http.server                # then open http://localhost:8000
```
Opening `index.html` directly also works, because the results are embedded. After re-running `train.py`, the site reads the fresh `results.json` when served over http (for example on GitHub Pages).

Publishing: GitHub repo > Settings > Pages > Deploy from a branch > `main` / `(root)`.

## Method in brief
- Features known at prediction time: store, week of year, holiday flag, temperature, fuel price, CPI, unemployment, and last week's sales of the same store (`lag1`).
- Split by time (first 70% of weeks train, next 15% validation, last 15% test), fixed before any model is compared. Scaling and one-hot encoding are inside a scikit-learn Pipeline, so they are fit on training rows only.
- Baselines: last week's sales; store average. Models: Ridge, Random forest, Gradient boosting (HistGradientBoostingRegressor).
- Hyperparameters and the final model are chosen on validation only. The test block is scored once per model, after refitting on train + validation.
- Metrics: MAE, RMSE, WMAPE. Extra analysis: holiday vs normal weeks, largest errors, permutation importance, and a bootstrap (over test weeks) interval for the gain over the naive rule.

## Main result (test block, 22 weeks, 990 rows)
Gradient boosting has MAE about $43,968 against $51,307 for the naive last-week rule (about 14% lower). The 95% bootstrap interval for the gain is wide ($124 to $15,120), so the improvement is real but modest. In holiday weeks the model is no better than the naive rule. These are predictive results only, not causal claims. Full tables and discussion are on the website.

## Interaction
Section 7 of the website is an interactive store explorer: pick a store, toggle the model and naive lines, and hover over a week to read actual, predicted and naive values and errors. Holiday weeks are shaded. It also reports in how many of the 45 stores the model beats the naive rule.

## Contributions
- Hamsika Para (23110230): problem framing, data understanding and plots, preprocessing and leakage checks, baselines, repository and GitHub Pages setup.
- Aakash Venkatesan (23110002): evaluation design, results and error analysis, bootstrap interval and permutation importance, the interactive store view.
- Both: README, references, testing the live site.
Section 12 of the website also gives key choices, one limitation each, and a small-change check (refitting without last week's sales, and without the macro features).

## Reproducibility notes
- Random seed 42 for models, importance and bootstrap. Results should match when using the versions in `requirements.txt`; other versions may change numbers slightly.
- Limitations are listed on the website (short history, single split, few holiday weeks, `lag1` only suits next-week forecasting).

## AI-use disclosure
We used AI to help draft `train.py`, `index.html` and this README and to discuss the evaluation design. All reported numbers are produced by running `train.py` on `Walmart.csv`; none were typed in by hand. We take responsibility for the final content and can explain and modify every part of it.

## References and citations
- Walmart sales dataset, Kaggle (same stores and weeks as the Walmart Recruiting - Store Sales Forecasting competition data).
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.
- McKinney (2010). Data Structures for Statistical Computing in Python. Proc. SciPy.
- Harris et al. (2020). Array programming with NumPy. Nature 585, 357-362.
- Hoerl and Kennard (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics 12(1), 55-67.
- Breiman (2001). Random forests. Machine Learning 45(1), 5-32.
- Friedman (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics 29(5), 1189-1232.
- Efron and Tibshirani (1993). An Introduction to the Bootstrap. Chapman and Hall.
