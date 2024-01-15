import pandas as pd
import json


def create_train_test(train_file_path, test_file_path, quantile=0.95):
    """
    Converts JSON files of train and test sets into dataframes.

    Args:
        train_file_path (str): The file path of the JSON file holding the train set
        test_file_path (str): The file path of the JSON file holding the test set
        quantile (float): the quantile to use of outliers removal

    Returns:
        tuple: the train and test dataframes
    """

    train_file = open(train_file_path)
    test_file = open(test_file_path)

    train_data = json.load(train_file)
    test_data = json.load(test_file)

    train = create_data(train_data)
    test = create_data(test_data)

    train_file.close()
    test_file.close()

    # remove outliers
    q = train[train['value'] > 0]['value'].quantile(quantile)
    train = train[train['value'] < q]
    test = test[test['value'] < q]

    return train, test

def reformat_pattern_names(df):
    """
    Converts the pattern names, used in the dataset's columns, into a more readable format

    Args:
        df (object): The dataframe with the pattern names as columns

    Returns:
        list os strings: the list of renames patterns
    """
    feature_names = []
    for feature_name in list(df.columns):
        if feature_name == 'value':
            feature_names.append(feature_name)
            continue
        feature_name = feature_name.replace('_', ' ')
        feature_name = feature_name.replace('@@Pair:', '')
        feature_name = feature_name.replace('@', ') ', 1)
        feature_name = feature_name.replace('@', ' (')
        feature_name = feature_name.replace(':', ': ')
        feature_name = feature_name.title()
        feature_name = "(" + feature_name + ")"
        feature_names.append(feature_name)
    return feature_names


def create_data(entities_list):
    """
    Converts the JSON objects in the dataset's JSON file to a dataframe
    with the distribution of medical patterns and a medication dosage value.

    Args:
        entities_list (list of object): An object with the distribution of patterns and a medical dosage vale

    Returns:
        object: The created dataframe
    """
    rows = []
    for entity_obj in entities_list:
        row = {}
        row['value'] = entity_obj['value']
        entity_patterns = entity_obj['entity_stats']
        for pattern_name in sorted(entity_patterns):
            value = entity_patterns[pattern_name]
            row[pattern_name] = value
        rows.append(row)
    raw_df = pd.DataFrame(rows)
    raw_df.columns = reformat_pattern_names(raw_df)
    return raw_df
