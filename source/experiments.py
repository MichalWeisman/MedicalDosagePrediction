import pandas as pd
import os
from .data_creation import create_train_test
from .two_step_regression import TwoStepRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor





def run_experiment_folds(experiment_path, folds):
    """
    Runs N experiments for a given parameter value in a medical domain.
    In each experiment, a Two-Step-Regression model is used.
    Three model configurations are tested at each run.
    The results are saved in a csv file in the parameter's value directory.

    Args:
        experiment_path (str): The directory of the specific parameter value in a medical domain.
        folds (int): The number of folds used for cross-validation
    """
    results = []
    for fold in range(1, folds + 1):
        print('Fold ', fold, '\\', folds, end='\r')
        fold_results_path = experiment_path + '\\evaluation'
        if not os.path.exists(fold_results_path):
            os.makedirs(fold_results_path)

        train_file_path = experiment_path + '\\' + str(fold) + '\\train.txt'
        test_file_path = experiment_path + '\\' + str(fold) + '\\test.txt'

        train, test = create_train_test(train_file_path, test_file_path)
        X_train = train.drop('value', axis=1)
        y_train = train['value']
        X_test = test.drop('value', axis=1)
        y_test = test['value']


        # models evaluations
        for configuration in ['all', 'two_step_pos', 'two_step_pos_classified']:
            clf = RandomForestClassifier(random_state=42)
            reg = GradientBoostingRegressor(random_state=42)
            model = TwoStepRegression(configuration, clf, reg)
            model.fit(X_train, y_train)
            result = model.evaluate(X_test, y_test)
            result['Configuration'] = configuration
            results.append(result)

    pd.DataFrame(results).to_csv(fold_results_path + '\\folds_results.csv', index=False)


def aggregate_folds(experiment_path):
    """
    Aggregates the results of the cross-validation runs into a single result table using mean values.
    Saves the table in a csv file.

    Args:
        experiment_path (str): The directory of the specific parameter value in a medical domain.
    """
    fold_results_dir = experiment_path + '\\evaluation'
    fold_results_path = fold_results_dir + '\\folds_results.csv'
    fold_results_df = pd.read_csv(fold_results_path)
    result = fold_results_df.groupby('Configuration').mean()
    result.to_csv(fold_results_dir + '\\aggregated_results.csv')
    return result

def run_experiment_for_parameter_value(experiment_path, folds):
    """
    Runs cross-validation experiments for a specific parameter value in a medical domain.
    Aggregates the results of all the folds and saves them in a csv file.

    Args:
        experiment_path (str): The directory of the specific parameter value in a medical domain.
        folds (int): The number of folds used for cross-validation

    """
    run_experiment_folds(experiment_path, folds)
    aggregate_folds(experiment_path)

def run_experiments_for_parameter(experiments_dir, domain_name, parameter_name, folds):
    """
    Runs a full experiment for all possible parameter values in a given medical domain
    Saves each results table in the corresponding parameter value directory.

    Args:
        experiments_dir (str): The data directory (which will also hold the results).
        domain_name (str): The name of the medical domain.
        parameter_name (str): The name of the parameter to be examined.
        folds (int): The number of folds used for cross-validation

    """
    parameter_path = experiments_dir + '\\' + domain_name + '\\' + parameter_name
    parameter_values = os.listdir(parameter_path)
    for parameter_value in parameter_values:
        print('Domain: ', domain_name, ', Parameter: ', parameter_name, ' = ', parameter_value)
        experiment_path = parameter_path + '\\' + parameter_value
        run_experiment_for_parameter_value(experiment_path, folds)

