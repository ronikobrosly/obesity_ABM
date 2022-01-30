import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



### READ IN AND DO BASIC MERGE

demo_df = pd.read_sas("~/Desktop/DEMO_J.XPT")
exam_df = pd.read_sas("~/Desktop/BMX_J.XPT")

demo_df = demo_df[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA', 'INDFMPIR']]
exam_df = exam_df[['SEQN', 'BMXBMI']]

merged_df = demo_df.merge(exam_df, how = "left", on = "SEQN")



### FIRST PASS AT CODING

# SEX
merged_df['MALE'] = np.nan
merged_df['MALE'][merged_df['RIAGENDR'] == 1] = 1
merged_df['MALE'][merged_df['RIAGENDR'] == 2] = 0

# AGE
merged_df['AGE'] = merged_df['RIDAGEYR']

# RACE_ETH
merged_df['RACE'] = np.nan
merged_df['RACE'][merged_df['RIDRETH3'] == 1] = 'HISPANIC'
merged_df['RACE'][merged_df['RIDRETH3'] == 3] = 'WHITE'
merged_df['RACE'][merged_df['RIDRETH3'] == 4] = 'BLACK'
merged_df['RACE'][merged_df['RIDRETH3'] == 6] = 'ASIAN'

# EDU
merged_df['EDU'] = np.nan
merged_df['EDU'][merged_df['DMDEDUC2'] < 2] = 'LS_HS'
merged_df['EDU'][merged_df['DMDEDUC2'].isin([3,4])] = 'HS'
merged_df['EDU'][merged_df['DMDEDUC2'] == 5] = 'COLLEGE'

# INCOME
merged_df['LOW_INCOME'] = np.nan
# merged_df['LOW_INCOME'][merged_df['INDFMIN2'].isin([1,2,3,4,5,6,7])] = 1
# merged_df['LOW_INCOME'][merged_df['INDFMIN2'].isin([8,9,10,14,15])] = 0
merged_df['LOW_INCOME'] = np.where(merged_df['INDFMPIR'] < merged_df['INDFMPIR'].median(), 1, 0)



# OBESITY
merged_df['OBESE'] = np.nan
merged_df['OBESE'][merged_df['BMXBMI'] < 30] = 0
merged_df['OBESE'][merged_df['BMXBMI'] >= 30] = 1

merged_df2 = merged_df[['SEQN', 'MALE', 'AGE', 'RACE', 'EDU', 'LOW_INCOME', 'OBESE', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA']]
merged_df2 = merged_df2.dropna()



#### PREP FOR LOG REG

merged_df2['HISPANIC'] = np.where(merged_df2['RACE'] == 'HISPANIC', 1, 0)
merged_df2['BLACK'] = np.where(merged_df2['RACE'] == 'BLACK', 1, 0)
merged_df2['ASIAN'] = np.where(merged_df2['RACE'] == 'ASIAN', 1, 0)

merged_df2['LS_HS'] = np.where(merged_df2['EDU'] == 'LS_HS', 1, 0)
merged_df2['HS'] = np.where(merged_df2['EDU'] == 'HS', 1, 0)

merged_df3 = merged_df2[['SEQN', 'MALE', 'AGE', 'HISPANIC', 'BLACK', 'ASIAN', 'LS_HS', 'HS', 'LOW_INCOME', 'OBESE', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA']]

merged_df3.to_csv("~/Desktop/transformed_NHANES.csv", index = False)
