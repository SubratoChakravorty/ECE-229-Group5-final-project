#!/usr/bin/env python
# coding: utf-8

# In[181]:


import numpy as np 
import pandas as pd
from scipy import stats


# In[196]:


school_file='hsls_school_v1_0.csv'
sc=pd.read_csv(school_file)  #read file


# In[186]:


school_varibale = sc[['X1CONTROL','X1LOCALE','X1REGION']]
school_varibale = school_varibale.astype({'X1REGION':'category','X1CONTROL':'category','X1LOCALE':'category' })
school_varibale.head()


# In[195]:


student_file='hsls_student_v1_0.csv'
st=pd.read_csv(student_file) #read file


# In[239]:


teacher_varibale = st[['N1SEX','X1TSCERT','N1GROUP','N1INTEREST','N1CONCEPTS','N1TERMS','S1STCHVALUES','S1STCHRESPCT','S1STCHFAIR','S1STCHCONF','S1STCHMISTKE','S1STCHTREAT']]
teacher_varibale = teacher_varibale.astype({'N1SEX':'category','X1TSCERT':'category','N1GROUP':'category'})
teacher_varibale.head()


# In[549]:


student_variable = st[['X1SEX','X1RACE','X1SCIEFF','X1SES','X1SCIID','X1SCIUTI','X1SCIINT','S1TEFRNDS','S1TEACTIV','S1TEPOPULAR','S1TEMAKEFUN']]
student_variable = student_variable.astype({'X1SEX':'category','X1RACE':'category'})
student_variable.head()


# In[562]:


def get_feature_importance(x,y):
    """
    - This function describes how important a particular feature(variable our user choose) is to predict y value(outcome our user expect to see)
    - parameter: x: two types of variable in our variables list, categorical type(gender or location) and numerical type(scale of something)
                 y: a dependent y field(outcome our user expect to see)
    - return:  It returns two sets of values:
               1.for categorical fields: returns statistical test results
               2.for numerical fields: returns pearson correlation coefficient between a field and y
    """
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    #assert y.dtypes[0] != 'category' # y must be continuous
    
    # if x is numerical(continuous) field, we return the pearson correlation between a field and y
    # for the pearson correlation between a field and y, their size must be the same 
    if x.dtypes[0] == 'float64' or 'int64':
        x = x.iloc[:,0] # to series
        y = y.iloc[:,0] # to series
        return stats.pearsonr(x, y)[0] # correlation coefficient
    else:
    # if x is categorical field, we return the statistical test results:
    # if x filed has 2 options like sex, we do the T-test(which is included by the ANOVA anlysis)
    # if x has more options like number of science courses, we do the ANOVA anlysis
        result = pd.concat([x,y],axis=1)
        df1 = [x for _, x in result.groupby(result[result.columns[0]])]
        data = []
        for i in range(1,len(df1)):
            data.append(df1[i][df1[i].columns[1]])
        return stats.f_oneway(*data)[1]
    return 'wrong data input'


# In[565]:


self_efficieny = student_variable[['X1SCIEFF']]
Socioeconomic_status = student_variable[['X1SES']]
get_feature_importance(self_efficieny, Socioeconomic_status)# return pearson correlation coefficient


# In[564]:


get_feature_importance(x,y) #return p_value 0.017421643110539238 compared to the p_value the following


# In[566]:


# use ANOVA test to get p_value
y = student_variable[['X1SCIEFF']] # continuous variable of self-efficiency
x = student_variable[['X1SEX']]
result = pd.concat([x,y],axis=1)
df1 = [x for _, x in result.groupby(result[result.columns[0]])]
a = []
for i in range(1,len(df1)):
    a.append(df1[i][df1[i].columns[1]])
stats.f_oneway(*a)[1]


# In[567]:


# use t-test to get p_value
cat1 = student_variable[student_variable['X1SEX']==1.0].X1SCIEFF
cat2 = student_variable[student_variable['X1SEX']==2.0].X1SCIEFF 
p_value = stats.stats.ttest_ind(cat1, cat2)[1]
p_value

