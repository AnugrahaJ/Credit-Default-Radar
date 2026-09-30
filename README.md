# Credit Default Radar — Loan Default Prediction with AI Credit Notes

> 🚧 **Status: In progress.** This README describes the project plan. 

A lender receives a loan application. **How likely is this applicant to default, why, and what should the lender do?**

This project builds a tool that answers all three. A machine learning model will estimate the probability of default, SHAP will explain which factors drove that estimate, and an LLM will turn the result into a short credit note a loan officer can read in seconds.

> **Core principle:** the model decides, the LLM only explains. The LLM will receive the probability, risk band and top SHAP reasons, and will be instructed not to invent reasons beyond them.

---

## Project Status

- [ ] 1. Data Cleaning
- [ ] 2. EDA
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
- **Size:** ~32,500 loan applications, 12 columns
- **Target:** `loan_status` (1 = default)
- **Features:** applicant profile (age, income, home ownership, employment length), loan details (purpose, amount, interest rate, loan-to-income), credit history (past default on file, credit history length), and the lender's own loan grade

**Limitations:** the data is simulated, amounts are in USD, and it lacks the signals real lenders rely on most, such as repayment history, bounced EMIs, salary regularity and bureau enquiries. The goal is to demonstrate the approach, not production-ready performance.

---

## Project Plan

### 1. Data Cleaning
- Check shape, default rate and duplicates
- Handle missing values in `person_emp_length` and `loan_int_rate`
- Remove impossible values, such as unrealistic ages and employment lengths
- Justify each cutoff and fill strategy in a cleaning log

### 2. EDA — framed as lender questions

| Lender question | What I'll investigate |
|---|---|
| Can they repay? | Is there a loan-to-income level where default suddenly spikes, and does burden predict default better than income alone? |
| Have they repaid before? | Do applicants with a past default default more now, and does a past default make a high-burden loan riskier? |
| What did the lender already think? | Does default rise from grade A to G, and within the same grade, does burden still change the risk? |
| What's the loan for? | Which purposes default the most, and does the interest rate rise to match that risk? |
| Data quality as a signal | Who are the applicants with missing employment details, and is missingness a warning sign in itself? |

Output: 5–6 findings in plain business language.

### 3. Feature Engineering & Modelling
- Build finance-style features (ratios and bands), encode categoricals, and use a stratified train/test split
- **Leakage test:** `loan_grade` and `loan_int_rate` encode the lender's own risk judgement, so models will be trained with and without them
- **Models:** logistic regression (baseline) and LightGBM (main model), with class imbalance handled
- **Metrics:** AUC, Gini and KS, the standard measures used in credit scoring
- **Threshold:** set from the relative cost of a bad loan versus a rejected good customer, not the default 0.5
- **Risk bands:** convert probabilities into low / medium / high risk, each with an action (approve / manual review / decline)

### 4. Explainability (SHAP)
- Global drivers of default across all applicants
- Individual explanations for 2–3 specific applicants
- A sanity check that the drivers make financial sense

### 5. AI Credit Note (Groq + LLaMA)
A function that takes an applicant's details, default probability, risk band and top SHAP reasons, and returns a short credit note with the risk level, main concerns, positive factors and suggested action. It will be tested on low, medium and high risk applicants.

### 6. API (FastAPI)
One endpoint that takes an applicant's details and returns the probability, risk band, top reasons and credit note. It will be a thin wrapper around the same pipeline function, so training and scoring use identical preprocessing.


## Planned Scoring Pipeline

```
Applicant details
   → same feature engineering + saved preprocessing
   → model → probability of default
   → saved cutoffs → risk band + action
   → SHAP → top risk-increasing and risk-reducing factors
   → reason dictionary → plain-language facts
   → LLM → credit note
   → response: probability, band, reasons, note
```

## How a Lender Would Use It

A loan officer submits an application and gets back a score, a band and a one-paragraph note. Low-risk applications move quickly, high-risk ones are declined with stated reasons, and the officer's time goes to the medium band, where judgement matters most.

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
