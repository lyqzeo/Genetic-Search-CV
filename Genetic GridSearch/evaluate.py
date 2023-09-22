import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import mean_absolute_percentage_error as mape_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score

class Evaluate:
    """
    Attributes
    ----------------
    p_type: problem type, reg/class

    
    """

    @classmethod
    def initialise(cls, **params):
        """
        To check if regression or classification

        Parameters
        --------------------
        params: for y
        thresh: Max no. of unique classes to accept in y for classification problem, dafault = 10

        Returns
        -------------------
        None
        """
        try:
            y = params['y']
        except KeyError:
            raise KeyError('Evaluate class was not initialised')
        

        problem_mapping = {
            'regression': ['float', 'int', 'int32', 'int64']  ,         ## Regression = 1
            'classification': ['int', 'str', 'bool', 'int32', 'int64']     ## Classification = 0
        }

        thresh = params.get('thresh', 10

        if y.nunique() == 1: ## Only 1 y value
            raise ValueError("Unable to model y as it only has 1 value")
        
        elif y.dtypes == 'float':
            print('Registering as a regression problem')
            cls.p_type = 1

        elif y.nunique() > thresh:
            print('Registering as a regression problem')
            cls.p_type = 1

        elif y.dtypes in problem_mapping['classification']:
            print('Registering as classification problem')
            cls.p_type = 0
        
        else:
            raise Exception('Data type of y not supported. Only "float"/"int" for regression, "str/int"(<=10 nunique) for classification')


    
    def __call__(self, actual, predicted, metric):
        # print(metric)
        '''Returns desired metrics'''
        try:
            self.p_type

        except AttributeError:
            raise AttributeError('Evaluate class was not initialised')


        if self.p_type == 0:    ## Classification Problem
            metrics = {
                        "accuracy": accuracy_score(actual, predicted),
                        "f1": f1_score(actual, predicted, average = "weighted", zero_division = 0),
                        "recall": recall_score(actual, predicted, average = "weighted", zero_division = 0),
                        "precision": precision_score(actual, predicted, average = "weighted", zero_division = 0)
                        }
        else:  ## Regression Problem
            metrics = {
                        "r2":r2_score(actual, predicted), 
                        "rmse":mse_score(actual, predicted)**0.5,
                        "mse":mse_score(actual, predicted), 
                        "mae":mae_score(actual, predicted)
                        }

        return metrics[metric]