import pandas as pd
import numpy as np

import pytest

import d6tjoin

def fake_2dfs_identical():
    df = pd.DataFrame({'a':range(10)})
    df['b'] = ['b']*5+['bb']*5
    return [df, df.copy()]

def fake_2dfs_1missing():
    df = pd.DataFrame({'a':range(10)})
    df['b'] = ['b']*5+['bb']*5
    return [df, df.copy().drop(['b'],1)]

def test_internals():
    dfs = fake_2dfs_identical()

    pdj = d6tjoin.Prejoin(dfs, print_only=False)
    assert pdj.keys is None and pdj.keysdf is None
    assert all([dfg.shape==dfs[0].shape for dfg in pdj.dfs])
    assert all([dfg.shape==(pdj.nrows, dfs[0].shape[1]) for dfg in pdj.dfshead])
    dfc = pdj.head()
    assert all([dfg.head().equals(dfc[idx]) for idx,dfg in enumerate(dfs)])
    dfc = pdj.head(10)
    assert all([dfg.head(10).equals(dfc[idx]) for idx,dfg in enumerate(dfs)])

    # single keys param
    cfg_keys = ['b']
    pdj = d6tjoin.Prejoin(dfs,keys=cfg_keys)
    assert pdj.keys == [['b','b']] and pdj.keysdf == [['b'],['b']]
    assert all([dfg.shape==dfs[0].shape for dfg in pdj.dfs])
    assert all([dfg.shape==(pdj.nrows, len(cfg_keys)) for dfg in pdj.dfshead])

    dfs[1] = dfs[1].rename(columns={'b': 'c'})
    with pytest.raises(KeyError, match='Columns missing'):
        pdj = d6tjoin.Prejoin(dfs, keys=['b'])

    # different keys for dfs
    pdj = d6tjoin.Prejoin(dfs,keys=[['b'],['c']])
    assert pdj.keys == [['b','c']] and pdj.keysdf == [['b'],['c']]
    assert all([dfg.shape==dfs[0].shape for dfg in pdj.dfs])
    assert all([dfg.shape==(pdj.nrows, 1) for dfg in pdj.dfshead])
    pdj = d6tjoin.Prejoin(dfs,keys=[['b','c']], keys_bydf=False)
    assert pdj.keys == [['b','c']] and pdj.keysdf == [['b'],['c']]

    # multi keys param
    dfs[0]['b1']=dfs[0]['b'];dfs[1]['c1']=dfs[1]['c'];
    pdj = d6tjoin.Prejoin(dfs,keys=[['b','b1'],['c','c1']])
    assert pdj.keys == [['b','c'],['b1','c1']] and pdj.keysdf == [['b','b1'],['c','c1']]
    assert all([dfg.shape==dfs[0].shape for dfg in pdj.dfs])
    assert all([dfg.shape==(pdj.nrows, 2) for dfg in pdj.dfshead])

    # joins with keys specified
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,keys=['b'], print_only=False)
    assert pdj.columns_common()==['b']
    assert pdj.columns_all()==['b']

    dfs[1] = dfs[1].rename(columns={'b': 'c'})
    pdj = d6tjoin.Prejoin(dfs,keys=[['b'],['c']], print_only=False)
    assert pdj.columns_all()==['b','c']


def test_pre_columns():
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.columns_common()==['a','b']
    assert pdj.columns_all()==['a','b']

    pdj.describe()
    assert pdj.shape() == {0: (10, 2), 1: (10, 2)}

    dfs = fake_2dfs_1missing()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.columns_common()==['a']
    assert pdj.columns_all()==['a','b']

def test_pre_describe():
    # describe_str
    chk = {'b': {'median': 1.5, 'min': 1.0, 'max': 2.0, 'nrecords': 10.0}}
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.describe_str()[0].to_dict(orient='index')==chk
    pdj = d6tjoin.Prejoin(dfs,keys=['b'],print_only=False)
    assert pdj.describe_str()[0].to_dict(orient='index')==chk

    # describe_str
    chk = {'a': {'nrecords': 10, 'unique': 10, 'nan': 0, 'unique rate': 1.0},
     'b': {'nrecords': 10, 'unique': 2, 'nan': 0, 'unique rate': 0.2}}
    pdj = d6tjoin.Prejoin(dfs,print_only=False)
    assert pdj.describe_data()[0].to_dict(orient='index')==chk
    pdj = d6tjoin.Prejoin(dfs,keys=['b'],print_only=False)
    assert pdj.describe_data()[0].to_dict(orient='index')==chk

def test_pre_data_match():
    dfs = fake_2dfs_identical()
    pdj = d6tjoin.Prejoin(dfs,print_only=False)

    dfc = {'__left__': {0: 'b'},
 '__right__': {0: 'b'},
 '__similarity__': {0: 1.0},
 '__left-sample__': {0: 'bb'},
 '__right-sample__': {0: 'bb'},
 '__left-nunique__': {0: 2},
 '__right-nunique__': {0: 2}}

    assert pd.DataFrame(dfc).equals(pdj.data_match())

    dfc = {0: {'__left__': 'a',
  '__right__': 'a',
  '__similarity__': 1.0,
  '__left-sample__': 0,
  '__right-sample__': 0,
  '__left-nunique__': 10,
  '__right-nunique__': 10},
 1: {'__left__': 'b',
  '__right__': 'b',
  '__similarity__': 1.0,
  '__left-sample__': 'bb',
  '__right-sample__': 'bb',
  '__left-nunique__': 2,
  '__right-nunique__': 2}}

    assert dfc==pdj.data_match(ignore_value_columns=False, max_unique_pct=1.0).to_dict(orient='index')






