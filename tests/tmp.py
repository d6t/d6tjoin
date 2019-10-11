import pandas as pd
import importlib

import d6tjoin
import d6tjoin.utils
importlib.reload(d6tjoin.utils)

df1=pd.DataFrame({'v':list(range(10))*2,'g':['a']*10+['b']*10})
df2=df1.copy()

j = d6tjoin.PreJoin([df1,df2])
j.str_describe()
j.data_describe()
j.columns_common()
j.columns_ispresent()
j.data_match()

j = d6tjoin.PreJoin([df1,df2], print_only=False)
r = j.data_match()
dfc = {'__left__': {0: 'g', 1: 'v'},
 '__right__': {0: 'g', 1: 'v'},
 '__similarity__': {0: 1.0, 1: 1.0}}
dfc = pd.DataFrame(dfc)
assert r.equals(dfc)
print(r)

quit()

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

