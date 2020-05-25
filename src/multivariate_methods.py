import pandas as pd

from .univariate_methods import load_data_frame


def get_correlation_matrix(fields, file_loc="../data/student_data.csv"):
    '''
    Computes correlation matrix that captures correlation between features
    presented in the fields parameter

    :param fields: List of fields
    :type fields: list
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: Correlation matrix.
    :rtype: pandas.DataFrame
    '''
    assert isinstance(fields, list), f"fields must be a list, not {type(fields)}"
    assert isinstance(file_loc, str), f"file_loc must be a string, not {type(file_loc)}"

    df = load_data_frame(file_loc)
    df_sub = df[fields]

    assert all([(isinstance(field, str) and field in df.columns) for field in fields])
    corrmat = df_sub.corr()

    return corrmat



# print(get_correlation_matrix(['X1SCIEFF','X1SES','X1SCIUTI','COST_PERCEPTION','TCH_PRCVD_ATT']))