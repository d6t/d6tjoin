import pandas as pd
import numpy as np

import d6tjoin

def fake_2dfs_identical():
    df = pd.DataFrame({'a':range(10)})
    df['b'] = ['b']*5+['bb']*5
    return [df, df.copy()]

def fake_2dfs_1missing():
    df = pd.DataFrame({'a':range(10)})
    df['b'] = ['b']*5+['bb']*5
    return [df, df.copy().drop(['b'],1)]

def test_pre_columns():
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.columns_common()==['a','b']
    assert pdj.columns_all()==['a','b']
    columns = ['b']
    # assert pdj.str_describe()==['a','b']
    # assert pdj.str_describe(columns=['b'])==['a','b']
    # assert pdj.str_describe(columns=['a'])==None

    dfs = fake_2dfs_1missing()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.columns_common()==['a']
    assert pdj.columns_all()==['a','b']

def test_keys_param():
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,keys=['b'], print_only=False)
    assert pdj.columns_common()==['b']
    assert pdj.columns_all()==['b']

    tdfs = [dfs[0],dfs[1].rename(columns={'b':'c'})]
    pdj = d6tjoin.Prejoin(tdfs,keys=[['b'],['c']], print_only=False)
    assert pdj.columns_all()==['b','c']


def dev_test_pre_strlen():
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.str_describe()==['a','b']
    assert pdj.str_describe(columns=['b'])==['a','b']
    assert pdj.str_describe(columns=['a'])==None
