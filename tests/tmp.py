import pandas as pd
import importlib

import d6tjoin
import d6tjoin.utils
importlib.reload(d6tjoin.utils)

df1=pd.DataFrame({'a':range(3),'b':range(3)})
df2=pd.DataFrame({'a':range(3),'c':range(3)})
df2=pd.DataFrame({'a':range(3),'b':range(3,6)})
df2=pd.DataFrame({'a':range(3,6),'c':range(3)})


j = d6tjoin.utils.BaseJoin([df1,df2],['a'])

j = d6tjoin.utils.BaseJoin([df1,df2],['a','b'])
j.keys
dfr = j.stats_prejoin(return_results=True)
dfr
(~dfr['all matched']).all()

j = d6tjoin.utils.BaseJoin([df1,df2],['a'])
j.stats_prejoin(return_results=True).to_dict()

