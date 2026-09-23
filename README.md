Financial Fraud Detection with Machine Learning

An end-to-end fraud detection pipeline built on 6.3 million simulated mobile money transactions, where only 0.13% of transactions are fraudulent — an extreme class imbalance problem.

Overview

Mobile money fraud is rare but costly, and naive models tend to either miss it entirely or flag far too many legitimate transactions. This project builds a full pipeline — from exploratory analysis through a deployable model — that catches the large majority of fraudulent transactions while handling the severe class imbalance in the data.

Dataset
~6.3 million simulated mobile money transactions (PaySim-style synthetic dataset)
Only 0.13% of transactions are labeled fraudulent
Exploratory analysis showed fraud occurs exclusively in TRANSFER and CASH_OUT transaction types, and that over 1.1 million of those transactions drain the sender's account balance entirely — a common signature of account-draining fraud
Approach

EDA - identified which transaction types can be fraudulent and which structural patterns (e.g. zeroed-out sender balances) correlate with fraud

Preprocessing - built a scikit-learn Pipeline with a ColumnTransformer handling feature scaling (numeric) and one-hot encoding (categorical)

Modeling - trained a class-weighted Logistic Regression to counteract the ~1:770 fraud-to-legitimate ratio without resampling the data

Evaluation - evaluated on a held-out test set of 1.9 million records

Export - serialized the full pipeline with joblib so it can be reloaded and applied to new transactions without retraining

Results

94% recall on fraudulent transactions in the held-out test set (2,321 of 2,464 fraud cases caught)

Trained and evaluated on a strict train/test split with no data leakage between sets
