from src.univariate_methods import load_data_frame
from src.univariate_methods import get_var_info
import src.config as config
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats

np.random.seed(0)


def get_feature_importance(y, fields, file_loc=config.student_data_file, method='pearson'):
    """
    Computes feature importance of all the variables in fields parameter with dependent y variable. For categorical
    fields, it provides results from anova analysis and for numerical/continuous fields it returns correlation
    coefficients with y.
    
    :param method: Method of correlation
    - pearson : standard correlation coefficient
    - kendall : Kendall Tau correlation coefficient
    - spearman : Spearman rank correlation
    :param file_loc: path to dataset
    :type file_loc: str
    :param fields: list of field ids where each field id is a string
    :type fields: list
    :param y: a dependent y field id
    :type y: str
    :return: a multilevel dictionary, value corresponding to key 'category' contains a dictionary with anova results for
    categorical fields and value for key 'continuous' is dictionary with correlation coefficients.
    :rtype dict
    """

    df = load_data_frame(file_loc)
    var_info = get_var_info()

    assert isinstance(fields, list)
    assert all([(isinstance(field, str) and field in var_info.index) for field in fields])
    assert isinstance(y, str)
    assert y in var_info.index
    assert isinstance(file_loc, str)

    # if x is numerical(continuous) field, we return the pearson correlation between a field and y
    # for the pearson correlation between a field and y, their size must be the same
    res = dict()
    res['continuous'] = dict()
    res['categorical'] = dict()
    for x in fields:
        if var_info.loc[x]['type'] == 'continuous':
            res['continuous'][x] = df[x].corr(df[y], method=method)
            print(method)
        elif var_info.loc[x]['type'] == 'categorical':
            df_sub = df[[x, y]].dropna()
            data = [x for _, x in df_sub.groupby(by=x)[y]]
            res['categorical'][x] = tuple(stats.f_oneway(*data))

    return res


def get_correlation_matrix(fields, file_loc=config.student_data_file):
    """
    Computes correlation matrix that captures correlation between features
    present in the fields parameter

    :param fields: List of fields
    :type fields: list
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: Correlation matrix.
    :rtype: pandas.DataFrame
    """
    assert isinstance(fields, list), f"fields must be a list, not {type(fields)}"
    assert isinstance(file_loc, str), f"file_loc must be a string, not {type(file_loc)}"

    df = load_data_frame(file_loc)
    df_sub = df[fields]

    assert all([(isinstance(field, str) and field in df.columns) for field in fields])
    corrmat = df_sub.corr()

    return corrmat


class MLmodel:

    def __init__(self, file_loc=config.student_data_file):

        self.clf = None
        self.fields = None
        self.df = load_data_frame(file_loc)
        self.var_info = get_var_info()
        self.trained = False
        self.cat_cols = None
        self.cont_cols = None

    def train_model(self, y, fields, regressor=None, test_split=0):
        """
        train a machine learning model with y as dependent variable and variables in fields parameter as independent variables
        :param regressor: sklearn regressor object, if None default RandomForestRegressor is used
        :param test_split: float, if non-zero, train-test split is performed and training and test accuracy is returned else
        model trained on complete data and training accuracy along with -1 in place of test accuracy returned.
        :param y: string, dependent variable, should be numerical/continuous
        :param fields: list,  list of independent variables, can be numerical or categorical
        :return: returns a tuple with training accuracy and test accuracy if test_split > 0 else a tuple with training accuracy
        and -1 in place of test accuracy.
        """
        assert isinstance(fields, list)
        assert all([(isinstance(field, str) and field in self.var_info.index) for field in fields])
        assert y in self.var_info.index
        assert 0 <= test_split <= 1

        df_sub = self.df[fields + [y]]
        df_sub = df_sub.dropna()

        self.fields = fields
        Y = df_sub[y]
        df_sub = df_sub[fields]

        self.cat_cols = [field for field in fields if self.var_info.loc[field]['type'] == 'categorical']
        self.cont_cols = [field for field in fields if self.var_info.loc[field]['type'] == 'continuous']
        # print(f'Cat: {self.cat_cols} and cont: {self.cont_cols}')

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.cont_cols),
                ('cat', categorical_transformer, self.cat_cols)])

        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        if regressor is None:
            regressor = RandomForestRegressor()
        self.clf = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', regressor)])

        if test_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(df_sub, Y, test_size=0.2)
            self.clf.fit(X_train, y_train)
            self.trained = True
            return self.clf.score(X_train, y_train), self.clf.score(X_test, y_test)
        else:
            self.clf.fit(df_sub, Y)
            self.trained = True
            return self.clf.score(df_sub, Y), -1

    def predict_model(self, input_data):
        """
        returns model's prediction for the input_data
        :param input_data: dict, a dictionary with fields as keys and a scalar value or a list of values for each field,
        depending upon the number of samples
        :return: returns a 1-d numpy array with predicted y value for each sample
        """

        assert isinstance(input_data, dict)
        assert all([field in self.fields for field in input_data.keys()])
        assert len(self.fields) == len(input_data.keys())

        if self.trained:
            test_data = pd.DataFrame()
            for field in self.fields:
                if isinstance(input_data[field], (list, tuple, np.ndarray)):
                    test_data[field] = input_data[field]
                else:
                    test_data[field] = [input_data[field]]
            return self.clf.predict(test_data)
        else:
            raise Exception("Model not trained")

