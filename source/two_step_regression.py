from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


class TwoStepRegression(BaseEstimator):
    """
    A class for the model which predicts the medication dosage which should be given to a patients in the next time window,
    based on the distribution of clinical patterns.

    The model consists of two steps:
        First step: Using a binary classifying to predict whether a medical treatment should be given
        Second step: Using a regression model to predict the medication dosage only for positively classified instances.
                     for negatively classified instances, the model will return the value of zero.
        This separation is crucial for predicting non-treatment - when no treatment should be given,
         the returned value should be zero, and not a value very close to zero.

    Attributes:
        binary_model (object): A scikit-learn compatible classifier
        regression_model (object): A scikit-learn compatible regression model
        configuration (str): The type of data to use for the training of the second step (the regression model):
                                1. 'all' - Use all data.
                                2. 'two_step_pos' - Use only positive training data.
                                3. 'two_step_pos_classified' - Use only data which was positively
                                 classified in the first step.
    """
    def __init__(self, configuration, binary_model, regression_model):
        self.binary_model = binary_model
        self.regression_model = regression_model
        self.configuration = configuration

    def get_params(self):
        return {'configuration': self.configuration}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_train, y_train):
        """
        Trains a two-step-regression model, according to the given configuration.

        Args:
            X_train (object): A dataframe of patterns distribution
            y_train (list): A list of given medical dosage values

        Returns:
            object: The trained TwoStepRegression object
        """
        # get standard deviation of the data (will be used for evaluation)
        self.sd = np.std(y_train[y_train > 0])

        # fit a binary_model (convert y to binary)
        y_binary = y_train.apply(lambda value: True if value > 0 else False)
        self.binary_model.fit(X_train, y_binary)

        # fit a regression_model
        if self.configuration == 'two_step_pos': # use only positive data
            X_only_positives = X_train[y_train > 0]
            y_only_positives = y_train[y_train > 0]
            self.regression_model.fit(X_only_positives, y_only_positives)

        elif self.configuration == 'two_step_pos_classified': # use only data positively classified in the first step
            predictions = self.binary_model.predict(X_train)
            pos_predictions_indices = [i for i, x in enumerate(predictions) if x == True]
            X_only_positively_classified = X_train.iloc[pos_predictions_indices]
            y_only_positively_classified = y_train.iloc[pos_predictions_indices]
            self.regression_model.fit(X_only_positively_classified, y_only_positively_classified)

        elif self.configuration == 'all': # use all data
            self.regression_model.fit(X_train, y_train)

        return self

    def predict(self, X_test):
        """
         Predicts the medication dosage, based on a given pattern distribution.

         Args:
             X_train (object): A dataframe of patterns distribution

         Returns:
             list of doubles: the list of the predicted medical dosage values
         """
        binary_y_pred = self.binary_model.predict(X_test)
        binary_y_pred = binary_y_pred.astype('int')
        regression_y_pred = self.regression_model.predict(X_test)

        # combine the first and second steps by nullifying
        # the values predicted by the regression model (second step)
        # where the classifier (first step) returned a negative result
        return np.multiply(binary_y_pred, regression_y_pred).tolist()

    def evaluate(self, X_test, y_test):
        """
           Evaluates the model on a test set.

           Args:
               X_test (object): A dataframe of patterns distribution
               y_test (list): A list of given medical dosage values

           Returns:
               object: a dictionary of the computed metrics: AUC, R^2, MAE, RMSE,
               NMAE (normalized MAE), NRMSE (normzlied RMSE), TNR, FPR, FNR, TPR.
           """
        y_test_binary = [1 if pred > 0 else 0 for pred in y_test]
        auc_score = roc_auc_score(y_test_binary, self.binary_model.predict_proba(X_test)[:, 1])
        y_pred_binary = self.binary_model.predict(X_test)
        tnr, fpr, fnr, tpr = confusion_matrix(y_test_binary, y_pred_binary, normalize='true').ravel()

        y_pred_two_step = self.predict(X_test)

        positive_y_pred_indices = [i for i, x in enumerate(y_pred_binary) if x == 1]
        positive_two_step_y_pred = np.array(y_pred_two_step)[positive_y_pred_indices].tolist()
        positively_predicted_y_test = np.array(y_test)[positive_y_pred_indices].tolist()

        mae = mean_absolute_error(positively_predicted_y_test, positive_two_step_y_pred)
        rmse = np.sqrt(mean_squared_error(positively_predicted_y_test, positive_two_step_y_pred))

        nmae = mae / self.sd if self.sd > 0 else mae
        nrmse = rmse / self.sd if self.sd > 0 else rmse

        r_2 = r2_score(positively_predicted_y_test, positive_two_step_y_pred)

        return {
            'AUC': auc_score,
            'R^2': r_2,
            'MAE': mae,
            'RMSE': rmse,
            'NMAE': nmae,
            'NRMSE': nrmse,
            'TNR': tnr,
            'FPR': fpr,
            'FNR': fnr,
            'TPR': tpr
        }
