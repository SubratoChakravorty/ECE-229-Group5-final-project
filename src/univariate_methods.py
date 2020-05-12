import pandas as pd


def return_fields(file_loc="../data/student_data.csv"):
    '''
    returns the field values from the dataset
    :return:
    '''

    assert isinstance(file_loc, str)

    df = pd.read_csv(file_loc)

    val_details = {'STU_ID':'Student ID', 'X1RACE': 'Student Race', 'X1SEX': 'Student Sex', 'X1SES':'Socioeconomic status',
           'X1SCIEFF':'Student Science Self-efficacy', 'N1COURSE': 'Science Course', 'X1SCIID': 'Scale of student\'s science identity',
           'X1SCIUTI':'Scale of student\'s science utility', 'X1SCIINT': 'Scale of student\'s interest in fall 2009 science course',
           'S1TEFRNDS': 'Time/effort in math/science means not enough time with friends', 'S1TEACTIV':'Time/effort in math/science means not enough time \
            for extracurriculars', 'S1TEPOPULAR':'Time/effort in math/science means 9th grader won\'t be popular', 'S1TEMAKEFUN':'Time/effort in math/science\
             means people will make fun of 9th grader','X1CONTROL':'School Control', 'X1LOCALE': 'School Locale (Urbanicity)',
            'N1SEX': 'Science Teacher’s Sex', 'X1TSRACE': 'Science Teacher’s Race', 'X1TSCERT': 'Science teacher\'s science teaching certification',
           'N1HIDEG':'Science teacher\'s highest degree', 'N1SCIJOB': 'Science teacher held science-related prior to becoming a teacher',
           'N1ALTCERT': 'Science teacher entered profession through alternative certification program', 'N1SCIYRS912': 'Years science teacher has taught high school science',
           'N1GROUP': 'Science teacher has students work in small groups', 'N1INTEREST': 'increasing students\' interest in science',
           'N1CONCEPTS':'teaching basic science concepts', 'N1TERMS': 'important science terms/facts N1SKILLS science process/inquiry skills',
            'S1STCHVALUES': '‘9th grader\'s fall 2009 science teacher values/listens to students\' ideas', 'S1STCHRESPCT': '9th grader\'s fall 2009 science\
            teacher treats students with respect', 'S1STCHFAIR': '9th grader\'s fall 2009 science teacher treats every\
            student fairly', 'S1STCHCONF': '‘9th grader\'s fall 09 science teacher thinks all students can be successful',
           'S1STCHMISTKE': '9th grader\'s fall 09 science teacher think mistakes OK if students learn', 'X3TGPAENG': 'English GPA',
           'X3TGPAMAT' : 'Mathematics GPA', 'X3TGPASCI': 'Science GPA'}

    res = dict()
    for k, v in val_details.items():
        if k in df.columns:
            res[k] = v

    return res


def get_counts(field_name='', file_loc="../data/student_data.csv"):
    '''
    returns frequency counts of the input field from the dataframe
    :param file_loc: path to the csv file
    :param field_name: string, name of the field
    :return: returns a dictionary with frequency distribution
    '''

    assert isinstance(field_name, str)

    df = pd.read_csv(file_loc)

    assert field_name in df.columns
    field_data = df[field_name]

    return field_data.value_counts()


print(get_counts('X1SEX', file_loc='../data/student_data.csv'))






# print(return_fields())

