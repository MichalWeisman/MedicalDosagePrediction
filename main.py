from source.experiments import run_experiments_for_parameter

"""
Runs experiments for each medical domain and each parameter value.
For each medical domain, several parameter types are examined.
Each parameter type has multiple possible values.
Each parameter value of a given domain has five datasets (one for each split in a 5-fold cross validation).
A model is trained and evaluated for each fold, and all folds of a given parameter value are then aggregated.
A final aggregated result is saved in each parameter's directory
"""
if __name__ == "__main__":
    experiments_dir = 'data'
    folds = 5
    for domain_name in ['hypoglycemia']: #originally: ['hypoglycemia', 'hypokalemia', 'hypotension']
        for parameter_name in ['x_width', 'y_width']: #originally: ['x_width', 'y_width', 'prediciton_gap', 'statistics_type']
            run_experiments_for_parameter(experiments_dir, domain_name, parameter_name, folds)