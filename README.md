# Medical Dosage Prediction Model

## Overview

This project presents a predictive model designed to forecast medical treatment dosage based on the distribution of clinical data patterns within various medical domains. The model utilizes data extracted from the MIMIC-IV dataset, and the complete methodology can be found in the original paper by Weisman and Shahar [1].

[1] Weisman Raymond M, Shahar Y. Provision of Decision Support Through Continuous Prediction of Recurring Clinical Actions. Stud Health Technol Inform. 2023 Jun 29;305:200-203. doi: 10.3233/SHTI230462. PMID: 37386996.

Read the full paper here: [Continuous Prediction of Recurring Clinical Actions](https://ebooks.iospress.nl/pdf/doi/10.3233/SHTI230462)

## Data

The clinical data patterns were extracted in a separate project with different configurations, including:

- **Feature Window Size (x_width):** The window from which clinical data is collected to predict medical dosage in the future window.
- **Target Window Size (y_width):** The window in which the medical dosage should be predicted.

The distribution of patterns among feature windows is stored in the data directory, organized by medical domains and configurations. (For simplicity, the example includes only the Hypoglycemia domain). The medical domain directory includes sub-directories for various configurations, such as x_width of 12 and 24 hours (with a fixed y_width of 4 hours), and y_width of 2 and 4 hours (with a fixed x_width of 24 hours). Each parameter value consists of five directories for train-test data splits, for a 5-fold-cross validation.

(Other configurations were excluded for simplicity purposes.)

## Model

The model uses a Two-Step-Regression approach:

1. **First Step:** A binary classifier predicts whether a medical treatment should be administered.
2. **Second Step:** A regression model predicts medication dosage only for positively classified instances. For negatively classified instances, the model returns a value of zero. This distinction is crucial for predicting non-treatment, ensuring that when no treatment is warranted, the returned value is precisely zero, not just a value very close to zero to prevent a false positive rate (FPR) of 1.0.

Each model evaluation consists of three training strategies (configurations). Each strategy defines the data on which the second step (the regression model) will be trained on:
5. 'all' - Use all data.
6. 'two_step_pos' - Use only positive training data.
7. 'two_step_pos_classified' - Use only data which was positively classified in the first step.

## Output
Two evaluation tables are saved in a CSV file for each parameter value within each medical domain:
1. folds_results.csv - Stores the results for each fold and each training strategy of a specific parameter value.
2. aggregated_results.csv - Stores an aggregated (mean) result of all folds. This table provides an overview of the model's performance.