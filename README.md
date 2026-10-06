# Credit Default Radar — Loan Default Prediction with AI Credit Notes

A lender receives a loan application. **How likely is this applicant to default, why, and what should the lender do?**

This project builds a tool that answers all three. A machine learning model will estimate the probability of default, SHAP will explain which factors drove that estimate, and an LLM will turn the result into a short credit note a loan officer can read in seconds.

> **Core principle:** the model decides, the LLM only explains. The LLM will receive the probability, risk band and top SHAP reasons, and will be instructed not to invent reasons beyond them.

---

## Project Status

- [x] 1. Data Cleaning
- [x] 2. EDA
- [ ] 3. Feature Engineering & Modelling
- [ ] 4. Explainability (SHAP)
- [ ] 5. AI Credit Note (Groq + LLaMA)
- [ ] 6. API (FastAPI)
- [ ] 7. PySpark Replication
- [ ] 8. Final README & Results

---

## Business Problem

Lending errors are not equal. Approving an applicant who defaults loses the principal. Rejecting an applicant who would have repaid loses the interest income and the customer. A useful credit model has to do more than rank applicants: it needs a decision threshold that reflects this cost asymmetry, and explanations a loan officer can trust and act on.

## Data

- **Source:** [Credit Risk Dataset (laotse) on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Size:** 32,581 loan applications, 12 columns (32,409 after cleaning)
- **Target:** `loan_status` (1 = default)
- **Features:** applicant profile (age, income, home ownership, employment length), loan details (purpose, amount, interest rate, loan-to-income), credit history (past default on file, credit history length), and the lender's own loan grade

---

## 1. Data Cleaning ✅

| Issue | Decision | Reason |
|---|---|---|
| 165 exact duplicate rows | Dropped | Repeated records would overweight those applicants |
| Unrealistic ages | Kept age ≤ 95 | Removes impossible values (e.g. 144) while keeping plausible older applicants |
| Unrealistic employment length | Kept ≤ 50 years | Longer careers than this are not credible |
| Extreme incomes (up to ~$2M) | Kept | Rare but plausible; not errors |
| Missing employment length (887 rows) | Filled with median (4 years) **and** added an `emp_missing` flag | Missingness was informative: these applicants defaulted more (32% vs 22%) |
| Missing interest rate (3,094 rows) | Filled with the median rate **within each loan grade** | Missingness was not informative, and rates are set by grade |

Default rate barely moved (21.82% → 21.87%), so cleaning did not distort the target.

---

## 2. EDA — Key Findings ✅

I organised the analysis around four questions a lender asks: **Can they repay? Have they repaid before? What is the loan for? What did the lender already think?** For each, I wrote down my expectation before looking at the data. Overall default rate: **21.9%**.

### Finding 1: Loan burden has a sharp cliff at 30% of income

| Loan as % of income | Default rate | Applicants |
|---|---|---|
| 0–10% | 11.8% | 10,426 |
| 10–20% | 15.1% | 11,999 |
| 20–30% | 22.0% | 6,166 |
| 30–40% | 68.9% | 2,700 |
| 40–50% | 72.9% | 872 |
| 50%+ | 78.5% | 246 |

Below 30%, about 2 in 10 applicants default; above it, about 7 in 10. But a "reject above 30%" rule would still miss **62% of defaulters**, who borrow modest amounts and look safe on this measure.

### Finding 2: Income matters too, independently of burden
Default falls from **43% in the lowest income group to 9% in the highest**, gradually, with no cliff. Within every income group, crossing 30% burden still raises default, even among the highest earners (19% → 40%). **Income tells you who is fragile; burden tells you when a loan is too large.**

### Finding 3: A past default doubles risk, mostly where burden looks safe
Past defaulters default at **37.9% vs 18.4%**. But 62% of them repaid this time, and 69% of all defaulters had no past default. A past default adds the most risk **below** the 30% line (+17 to +25 points) and little above it (+6 points). **Use it as a review flag, not an automatic rejection.**

### Finding 4: The lender's grade ranks risk well but ignores burden
Default rises from **10% in grade A to 98% in grade G**, with a cliff between C (21%) and D (59%). But **a grade A applicant with 30–40% burden defaults at 59%**, worse than a grade D applicant with 0–10% burden (51%), while being charged the lowest rate (~7%). High-burden applicants in grades A–C are underpriced.

### Finding 5: Loan purpose changes risk, but every purpose is priced the same
Default ranges from **15% (venture) to 29% (debt consolidation)**, yet every purpose is charged about **11%**, a spread of only 0.3 points. Safer borrowers subsidise riskier ones, which over time can drive safe borrowers away (adverse selection).

### Finding 6: Missing employment details signal low income, not extra risk
Applicants with missing employment length default more (31.7% vs 21.6%). They are not younger (both average ~27) but poorer (median income **$36K vs $56K**). Within income groups the gap mostly disappears. The `emp_missing` flag is kept, and Stage 4 will check whether the model uses it.

### How the findings connect
Every signal catches part of the risk and misses the rest. Burden is strongest but misses most defaulters, past default finds some hidden risk, the grade ignores burden and purpose, and purpose adds a small unpriced signal. **No single rule is enough**, which is why the next stage combines them in a model.

---

## 3. Feature Engineering & Modelling (next)
- Finance-style features, encoding, and a stratified train/test split
- Imputation medians recomputed on **training data only** (cleaning used the full data, a minor leakage to fix here)
- **Leakage test:** `loan_grade` and `loan_int_rate` encode the lender's own risk judgement, so models will be trained with and without them
- **Models:** logistic regression (baseline) and LightGBM (main model), with class imbalance handled
- **Metrics:** AUC, Gini and KS
- **Threshold:** set from the cost of a bad loan versus a rejected good customer, not the default 0.5
- **Risk bands:** low / medium / high, each with an action (approve / manual review / decline)

## 4. Explainability — SHAP (planned)
Global drivers of default, individual explanations for 2–3 applicants, and a check that the drivers make financial sense.

## 5. AI Credit Note — Groq + LLaMA (planned)
A function that takes an applicant's details, default probability, risk band and top SHAP reasons, and returns a short credit note with risk level, main concerns, positive factors and suggested action.

## 6. API — FastAPI (planned)
One endpoint that takes an applicant's details and returns the probability, risk band, top reasons and credit note.

## 7. PySpark Replication (planned)
Cleaning, EDA and feature engineering rebuilt in PySpark to learn the Spark workflow. The dataset is small enough for pandas, so this is for learning, not scale.

---

## How a Lender Would Use It

A loan officer submits an application and gets back a score, a band and a one-paragraph note. Low-risk applications move quickly, high-risk ones are declined with stated reasons, and the officer's time goes to the medium band, where judgement matters most.

## Limitations

- The data is simulated, so some patterns may reflect how it was generated rather than real borrower behaviour.
- "Default" is not defined in the data; real lenders typically use 90+ days past due.
- Loan-to-income uses annual income and has no loan tenure, so it is a rough affordability measure, not a true EMI-to-income ratio (FOIR).
- Credit history is limited to two columns; a real bureau report has far more detail.
- Some combinations (grade G, high-burden grades E–F) have too few applicants to trust.

## What I'd Add With Real Data

- Repayment history and days past due (DPD) on existing loans
- Bounced EMIs and salary credit regularity from bank statements
- Recent bureau enquiries (credit hunger)
- Monitoring for drift after deployment (e.g. PSI on score distributions)

---

## Tech Stack

Python · pandas · scikit-learn · LightGBM · SHAP · Groq (LLaMA) · FastAPI · PySpark

---

**Author:** Anugraha Jayakumar · MSc Economics
