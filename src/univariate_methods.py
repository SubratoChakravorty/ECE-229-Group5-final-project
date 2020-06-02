from functools import lru_cache
import json
import math
import os
from typing import Union, Tuple

import pandas as pd
import src.config as config


def return_fields(file_loc=config.student_data_file):
    '''
    Returns the field values from the dataset

    :param file_loc: Path to the csv file
    :type file_loc: str
    :returns: Key-value pair
    :rtype: dict
    '''

    assert isinstance(file_loc, str)

    df = pd.read_csv(file_loc)

    val_details = {'STU_ID': 'Student ID', 'X1RACE': 'Student Race', 'X1SEX': 'Student Sex',
                   'X1SES': 'Socioeconomic status',
                   'X1SCIEFF': 'Student Science Self-efficacy', 'N1COURSE': 'Science Course',
                   'X1SCIID': 'Scale of student\'s science identity',
                   'X1SCIUTI': 'Scale of student\'s science utility',
                   'X1SCIINT': 'Scale of student\'s interest in fall 2009 science course',
                   'S1TEFRNDS': 'Time/effort in math/science means not enough time with friends', 'S1TEACTIV': 'Time/effort in math/science means not enough time \
            for extracurriculars', 'S1TEPOPULAR': 'Time/effort in math/science means 9th grader won\'t be popular',
                   'S1TEMAKEFUN': 'Time/effort in math/science\
             means people will make fun of 9th grader', 'X1CONTROL': 'School Control',
                   'X1LOCALE': 'School Locale (Urbanicity)',
                   'N1SEX': 'Science Teacher’s Sex', 'X1TSRACE': 'Science Teacher’s Race',
                   'X1TSCERT': 'Science teacher\'s science teaching certification',
                   'N1HIDEG': 'Science teacher\'s highest degree',
                   'N1SCIJOB': 'Science teacher held science-related prior to becoming a teacher',
                   'N1ALTCERT': 'Science teacher entered profession through alternative certification program',
                   'N1SCIYRS912': 'Years science teacher has taught high school science',
                   'N1GROUP': 'Science teacher has students work in small groups',
                   'N1INTEREST': 'increasing students\' interest in science',
                   'N1CONCEPTS': 'teaching basic science concepts',
                   'N1TERMS': 'important science terms/facts N1SKILLS science process/inquiry skills',
                   'S1STCHVALUES': '‘9th grader\'s fall 2009 science teacher values/listens to students\' ideas',
                   'S1STCHRESPCT': '9th grader\'s fall 2009 science\
            teacher treats students with respect', 'S1STCHFAIR': '9th grader\'s fall 2009 science teacher treats every\
            student fairly',
                   'S1STCHCONF': '‘9th grader\'s fall 09 science teacher thinks all students can be successful',
                   'S1STCHMISTKE': '9th grader\'s fall 09 science teacher think mistakes OK if students learn',
                   'X3TGPAENG': 'English GPA',
                   'X3TGPAMAT': 'Mathematics GPA', 'X3TGPASCI': 'Science GPA'}

    res = dict()
    for k, v in val_details.items():
        if k in df.columns:
            res[k] = v

    return res


def get_counts(field_name='', file_loc=config.student_data_file):
    '''
    Returns frequency counts of the input field from the dataframe

    :param file_loc: Path to the csv file
    :type file_loc: str
    :param field_name: The name of the field
    :type field_name: str
    :returns: Frequency distribution
    :rtype: dict
    '''

    assert isinstance(field_name, str)

    df = pd.read_csv(file_loc)

    assert field_name in df.columns
    field_data = df[field_name]

    return field_data.value_counts()


@lru_cache(maxsize=20)
def get_field_data(field_name: Union[str, Tuple] = '', file_loc=config.student_data_file):
    '''
    Returns the input field data from the dataframe

    :param field_name: Field name
    :type field_name: str or list of str
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: Input field data.
    :rtype: pandas.Series
    '''
    assert not isinstance(field_name, list), "A sequence of fields must be passed as a tuple"
    if isinstance(field_name, tuple):
        for e in field_name:
            assert isinstance(e, str)
        field_name = list(field_name)
    else:
        assert isinstance(field_name, str)

    df = pd.read_csv(file_loc)

    field_data = df[field_name]

    return field_data


def get_binned_data(field_name='', width=10, file_loc=config.student_data_file):
    '''
    Returns the count of continuous data count seperated by range

    :param width:
    :param field_name: Field name
    :type field_name: str
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: Midnumber of range and the count of data in different range
    :rtype: dict
    '''

    assert isinstance(field_name, str)

    df = pd.read_csv(file_loc)

    assert field_name in df.columns
    field_data = df[field_name]
    Range = max(field_data) - min(field_data)
    bins_num = math.ceil(Range / width)
    bins = list(range(bins_num))  # * int(width)
    for i in range(len(bins)):
        bins[i] *= width

    cut = pd.cut(field_data, bins)
    cut_res = pd.value_counts(cut)
    res = {}
    res["range"] = list(map(lambda x: x.mid, cut_res.index))
    res["count"] = list(cut_res)
    return res


def get_hierarchical_data(fields, color_var='X1SCIEFF', file_loc=config.student_data_file) \
        -> Tuple[pd.DataFrame, float]:
    '''
    Returns a dataframe with mean and count of groups segregated using input fields
    
    :param fields: List of fields
    :type fields: list
    :param color_var: continuous y variable
    :type color_var: str
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: A dataframe with info to build a sunburst plot
    :rtype: pandas.DataFrame
    '''

    assert isinstance(fields, list), f"fields must be a list, not {type(fields)}"
    assert isinstance(color_var, str), f"color_var must be a string, not {type(color_var)}"

    df = load_data_frame(file_loc)
    df = df[fields + [color_var]]
    color_var_mean = df[color_var].mean()

    assert color_var in df.columns
    assert all([(isinstance(field, str) and field in df.columns) for field in fields])

    df = df.groupby(by=fields)

    flat_df = df.count().reset_index().rename(columns={color_var: 'count'})
    flat_df['mean'] = df.mean()[color_var].values
    return flat_df, color_var_mean


@lru_cache(maxsize=10)
def load_data_frame(file_loc=config.student_data_file) -> pd.DataFrame:
    """
    Used to store dataframes loaded from a csv.

    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: Data
    :rtype: pandas.DataFrame
    """
    assert os.path.isfile(file_loc), f"{file_loc} is not in path"

    return pd.read_csv(file_loc)


def get_var_group(group, file_loc=config.vargroup_file):
    """
    Return a list of variables of a certain group
    
    :param group: Group of the variable to get
    :type group: str
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: List of variables in the specific group
    :rtype: list
    """
    assert isinstance(file_loc, str)
    assert isinstance(group, str)

    with open(file_loc, "r") as f:
        content = json.load(f)

    assert group in content

    return content[group]


def get_var_info(file_loc=config.variables_file):
    """
    Get variable information
    Usage
    1. Single variable: get_var_info("N1ALTCERT")
    2. Batch variables: get_var_info(["N1ALTCERT", "N1COURSE"])
    
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: A pd.DataFrame associated with the variable or a
        subset of pd.DataFrame corresponds to each variable in name.
    :rtype: pandas.DataFrame
    """
    assert isinstance(file_loc, str)

    df = pd.read_csv(file_loc, index_col=0)

    # Multiple variables
    return df


def get_stats(field, file_loc=config.student_data_file, median=True):
    '''
    Returns min,median(mean) and max of a numerical field

    :param median: Returns median as the middle value if True, else mean
    :type median: bool
    :param field: variable name
    :type field: str
    :param file_loc: Path to the dataset
    :type file_loc: str
    :returns: tuple(min, max, median (mean))
    :rtype: tuple
    '''
    assert isinstance(field, str), f'field must be of type str, and not {type(field)}'
    assert isinstance(file_loc, str), f'file_loc must be of type str, and not {type(file_loc)}'

    df = load_data_frame(file_loc)
    assert field in df.columns

    minm = df[field].min()
    maxm = df[field].max()

    if median:
        mid = df[field].median()
    else:
        mid = df[field].mean()

    return minm, mid, maxm


def get_categories(field, file_loc=config.student_data_file) -> Tuple[int, dict]:
    '''
    'returns the most common category as int and dictionary with mapping from integers to categories.
    :param field: categorical field
    :type str
    :param file_loc: path to the dataset
    :type str
    :return: returns a tuple with an int and dictionary
    :type tuple
    '''

    assert isinstance(field, str), f'field must be of type str, and not {type(field)}'
    assert isinstance(file_loc, str), f'file_loc must be of type str, and not {type(file_loc)}'

    df = load_data_frame(file_loc)
    var_info = get_var_info(config.variables_file)
    assert field in var_info.index, 'Invalid field name'
    assert var_info.loc[field]['type'] == 'categorical', 'field column must have categorical data and not' \
                                                         ' continuous/numerical data'

    categories = df[field].value_counts().index
    mapp = dict()
    for id, val in enumerate(categories):
        mapp[id+1] = val

    return 1, mapp

