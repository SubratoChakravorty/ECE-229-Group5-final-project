import pandas as pd
import os

from univariate_methods import load_data_frame
from univariate_methods import get_var_info


def get_correlation_matrix(fields, file_loc="../data/student_data.csv"):
    '''
    computes correlation matrix that captures correlation between features preesent in the fields parameter
    :param fields: list, list of fields names
    :param file_loc: string, path to the csv data file
    :return: pd.DataFrame, correlation matrix
    '''
    assert isinstance(fields, list), f"fields must be a list, not {type(fields)}"
    assert isinstance(file_loc, str), f"file_loc must be a string, not {type(file_loc)}"

    df = load_data_frame(file_loc)
    df_sub = df[fields]

    assert all([(isinstance(field, str) and field in df.columns) for field in fields])
    corrmat = df_sub.corr()

    return corrmat


class MLmodel:

    def __init__(self, file_loc="../data/student_data.csv"):

        self.df = load_data_frame(file_loc)
        self.var_info = get_var_info()
        print(self.var_info.index)

    def train_model(self, y, fields, num_trees=100):
        '''
        train a machine learning model with y as dependent variable and variables in fields as independent variables
        :param num_trees: number of estimators in Random forest
        :param y: string, dependent variable
        :param fields: list,  list of independent variables
        :return: returns the model object
        '''

        assert all([(isinstance(field, str) and field in self.var_info.index) for field in fields])
        assert y in self.var_info.index

        df_sub = self.df[fields + [y]]
        df_sub = df_sub.dropna()
        Y = df_sub[y]
        df_sub = df_sub[fields]

        from sklearn.model_selection import train_test_split
        df_sub_train, df_sub_test, Y_train, Y_test = train_test_split(df_sub, Y,test_size=0.3)
        from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
        from sklearn.svm import LinearSVR

        rf = RandomForestRegressor(n_estimators=num_trees)
        rf.fit(df_sub_train, Y_train)

    def predict_model(self, input_data):
        '''

        :param input_data:
        :return:
        '''

# model = MLmodel()
# model.train_model(y='X1SCIEFF', fields=['X1SES','COST_PERCEPTION','TCH_PRCVD_ATT','X1SCIID','X1SCIINT','X1SCIUTI','X3TGPASCI'])

# print(get_correlation_matrix(['X1SCIEFF','X1SES','X1SCIUTI','COST_PERCEPTION','TCH_PRCVD_ATT']))