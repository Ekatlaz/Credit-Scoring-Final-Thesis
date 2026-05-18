# Interpretable Credit Scoring Pipeline

This repository contains the code and report for a credit scoring project based on anonymized credit application data close to real-world banking data. The main goal of the project is to build an interpretable machine learning pipeline for predicting the probability of default and explaining the obtained results.

The work focuses not only on predictive quality, but also on feature processing, model interpretability, and stability of results across different data splits.

## Project overview

The project includes the following stages:

- exploratory data analysis of anonymized credit data;
- processing of missing and abnormal values;
- generation of derived financial features;
- separation of numerical and categorical features;
- construction of a preprocessing pipeline;
- comparison of categorical encoding methods;
- training and comparison of several classification models;
- feature selection and model interpretation;
- stability analysis of models and feature importance;
- construction of an LLM-assisted explanation layer for textual interpretation of scoring results.

## Models

The following models were trained and compared:

- Logistic Regression;
- Decision Tree;
- LightGBM;
- MLP Small;
- MLP Deep.

The models were evaluated using the following metrics:

- ROC-AUC;
- Average Precision;
- LogLoss;
- Accuracy.

The best predictive quality was achieved by LightGBM. Logistic regression provided a strong interpretable baseline, while MLP Small and MLP Deep were used as additional nonlinear benchmarks.

## Interpretability

Several interpretation methods were used in the project:

- logistic regression coefficients;
- decision tree rules;
- permutation importance;
- SHAP values;
- partial dependence plots.

The analysis showed that the most important factors were mainly related to external scoring indicators, loan parameters, debt burden, and demographic characteristics.

## LLM-assisted explanation layer

An additional part of the project is an LLM-assisted explanation layer. The language model is not used as a separate scoring model and does not predict the probability of default. Instead, it transforms the output of the scoring model and SHAP-based local explanations into a clear textual summary.

This layer is intended to make model explanations easier to understand for a human user, such as a risk analyst or manager.
