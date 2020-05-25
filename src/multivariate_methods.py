import pandas as pd

from .univariate_methods import load_data_frame


def get_correlation_matrix(fields, file_loc="../data/student_data.csv"):
    '''
    computes correlation matrix that captures correlation between features preesent in the fields parameter
    :param fields: list of fields names, list
    :param file_loc: path to the csv data file, str
    :return:
    '''
    assert isinstance(fields, list), f"fields must be a list, not {type(fields)}"
    assert isinstance(file_loc, str), f"file_loc must be a string, not {type(file_loc)}"

    df = load_data_frame(file_loc)
    df_sub = df[fields]

    assert all([(isinstance(field, str) and field in df.columns) for field in fields])
    corrmat = df_sub.corr()

    return corrmat



# print(get_correlation_matrix(['X1SCIEFF','X1SES','X1SCIUTI','COST_PERCEPTION','TCH_PRCVD_ATT']))