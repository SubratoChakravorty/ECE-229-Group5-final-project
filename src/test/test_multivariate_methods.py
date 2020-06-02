from unittest import TestCase
import numpy as np
import pandas as pd


class TestMultivariateMethods(TestCase):
    def test_get_feature_importance(self):
        from src.multivariate_methods import get_feature_importance
        fields = ['X1GEN', 'X1SES']
        y = 'X1SCIEFF'
        res = get_feature_importance(y=y, fields=fields)
        assert isinstance(res, dict)

    def test_get_correlation_matrix(self):
        from src.multivariate_methods import get_correlation_matrix
        fields = ['COST_PERCEPTION','TCH_PRCVD_ATT','X1SCIID','X1SCIINT','X1SCIUTI','X1SES']
        corr_mat = get_correlation_matrix(fields=fields)
        assert isinstance(corr_mat, type(pd.DataFrame()))

    def test_train_model(self):
        from src.multivariate_methods import MLmodel
        from sklearn.linear_model import LinearRegression

        mlmodel = MLmodel()
        fields = ['COST_PERCEPTION', 'TCH_PRCVD_ATT', 'X1SCIID', 'X1SCIINT', 'X1SCIUTI', 'X1SES', 'X1GEN']
        res = mlmodel.train_model(y='X1SCIEFF', fields=fields, regressor=LinearRegression(), test_split=0.2)

        assert isinstance(res, tuple)

    def test_predict_model(self):
        from src.multivariate_methods import MLmodel
        from sklearn.linear_model import LinearRegression

        mlmodel = MLmodel()
        fields = ['COST_PERCEPTION', 'TCH_PRCVD_ATT', 'X1SCIID', 'X1SCIINT', 'X1SCIUTI', 'X1SES', 'X1GEN']
        mlmodel.train_model(y='X1SCIEFF', fields=fields, regressor=LinearRegression(), test_split=0.2)
        sample_data = {'COST_PERCEPTION': 2.25, 'TCH_PRCVD_ATT': 4.0, 'X1SCIID': 0.91, 'X1SCIINT': -0.23, 'X1SCIUTI':
                       -0.33, 'X1SES': 1.5644, 'X1GEN': 'Male'}
        res = mlmodel.predict_model(sample_data)

        assert isinstance(res, np.ndarray)
        assert isinstance(res[0], float)



