import pandas as pd
import numpy as np
import itertools
from faker import Faker
import importlib

import d6tjoin.top1
import d6tjoin.utils

importlib.reload(d6tjoin.top1)

# *******************************************************
# generate sample time series data with id and value
# *******************************************************
nobs = 10
f1 = Faker()
f1.seed(0)
uuid1 = [str(f1.uuid4()).split('-')[0] for _ in range(nobs)]
dates1 = pd.date_range('1/1/2010','1/1/2011')

df1 = pd.DataFrame(list(itertools.product(dates1,uuid1)),columns=['date','id'])
df1['val1']=np.round(np.random.sample(df1.shape[0]),3)

# create mismatch
df2 = df1.copy()
df2['id'] = df1['id'].str[1:-1]
df2['val2']=np.round(np.random.sample(df2.shape[0]),3)

d6tjoin.utils.PreJoin([df1,df2],['id','date']).stats_prejoin()

result = d6tjoin.top1.MergeTop1(df1.head(),df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'],exact_left_on=['date'],exact_right_on=['date']).merge()

print(result['top1']['id'].head(2))

print(result['merged'].head(2))
