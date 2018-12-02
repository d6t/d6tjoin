import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import importlib
import d6tjoin.top1
import jellyfish
from faker import Faker

import tests.test_smartjoin

def gen_df2_str():
    l1 = ['a', 'b']
    l2 = [l1[0], 'ba', 'cd']
    df1 = pd.DataFrame({'id':l1*4})
    df2 = pd.DataFrame({'id':l2*4})
    df1['v1']=range(df1.shape[0])
    df2['v2']=range(df2.shape[0])
    return df1, df2

def gen_df2_num():
    l1 = [1,2]
    l2 = [l1[0],1.1,1.2]
    df1 = pd.DataFrame({'id': l1 * 4})
    df2 = pd.DataFrame({'id': l2 * 4})
    return df1, df2


def test_top1_gen_candidates():

    def helper(df1, df2):

        dfr = d6tjoin.top1.MergeTop1Diff(df1, df2,'id','id',jellyfish.levenshtein_distance)._allpairs_candidates()
        assert dfr.shape==(4, 3)
        assert (dfr['__top1left__'].values[0]==df1['id'].values[0])
        assert np.all(dfr['__top1left__'].values[1:]==df1['id'].values[1])
        assert (dfr['__top1right__'].values[0]==df1['id'].values[0])
        assert (dfr['__top1right__']==df2['id'].values[1]).sum()==1
        assert (dfr['__top1right__']==df2['id'].values[2]).sum()==1
        assert (dfr['__matchtype__']=='exact').sum()==1
        assert (dfr['__matchtype__']=='top1 left').sum()==3

    df1, df2 = gen_df2_str()
    helper(df1, df2)

    df1, df2 = gen_df2_num()
    helper(df1, df2)


def test_top1_str():

    df1, df2 = gen_df2_str()

    r = d6tjoin.top1.MergeTop1Diff(df1, df2,'id','id',jellyfish.levenshtein_distance).merge()
    dfr = r['top1']
    assert dfr['__top1diff__'].min()==0
    assert dfr['__top1diff__'].max()==1
    assert dfr.shape==(3, 4)
    dfr = r['merged']
    assert dfr.shape==(48, 4)
    assert np.all(dfr.groupby('id').size().values==np.array([16, 32]))

    df1, df2 = tests.test_smartjoin.gen_multikey_complex(unmatched_date=False)
    r = d6tjoin.top1.MergeTop1Diff(df1, df2,'key','key',jellyfish.levenshtein_distance,['date'],['date']).merge()
    dfr = r['merged']
    assert dfr.shape==(18, 5)
    assert np.all(dfr.groupby(['date','key']).size().values==np.array([1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    df1.head()
    df1.merge(df2, on=['date','key']).head()
    dfr.head()

def test_top1_num():

    df1, df2 = tests.test_smartjoin.gen_multikey_complex(unmatched_date=True)
    r = d6tjoin.top1.MergeTop1Number(df1, df2,'date','date',is_keep_debug=True).merge()
    dfr = r['top1']
    assert dfr.shape==(4, 4)
    assert np.all(dfr.groupby('__matchtype__').size().values==np.array([2, 2]))
    assert dfr['__top1diff__'].dt.days.max()==2
    assert dfr['__top1diff__'].dt.days.min()==0

    df1, df2 = tests.test_smartjoin.gen_multikey_complex(unmatched_date=True)
    r = d6tjoin.top1.MergeTop1Number(df1, df2,'date','date',['key'],['key']).merge()
    dfr = r['merged']
    dfr.sort_values(['date','key'])
    r['top1'].sort_values(['__top1left__','key'])
    df1.sort_values(['key','date'])
    df2.sort_values(['key','date'])
    r['top1']

def test_top1_multi():

    df1, df2 = tests.test_smartjoin.gen_multikey_complex(unmatched_date=True)
    df2['key'] = 'Mr. '+df1['key']

    r = d6tjoin.top1.MergeTop1(df1, df2,['date','key'],['date','key']).merge()


    assert True


def test_top1_examples():
    import uuid
    import itertools

    # ******************************************
    # generate sample data
    # ******************************************
    nobs = 10
    # todo: set uuid seed
    # todo: only pick first 2 blocks
    f1 = Faker()
    f1.seed(0)
    uuid1 = [str(f1.uuid4()).split('-')[0] for _ in range(nobs)]
    dates1 = pd.date_range('1/1/2010', '1/1/2011')
    dates2 = pd.bdate_range('1/1/2010', '1/1/2011')  # business instead of calendar dates

    df1 = pd.DataFrame(list(itertools.product(uuid1, dates1)), columns=['id', 'date'])
    df1['v'] = np.random.sample(df1.shape[0])
    df2 = df1.copy()
    df2['id'] = df1['id'].str[1:-1]

    # r = d6tjoin.top1.MergeTop1Number(df1, df2, 'id', 'id', ['date'], ['date']).merge()
    # assert raises ValueError => should check it's a number to do number join

    # r = d6tjoin.top1.MergeTop1Diff(df1, df2, 'id', 'id', jellyfish.levenshtein_distance, ['date'], ['date']).merge()
    # assert min()==2
    # assert diff no duplicates
    # assert diff found == substring
    # assert only 100 candidates (not 366*100)

    # r = d6tjoin.top1.MergeTop1(df1, df2, ['id'], ['id'], ['date'], ['date']).merge()
    # assert merged==merged
    # assert diff==diff

    # dates2 = pd.bdate_range('1/1/2010', '1/1/2011')  # business instead of calendar dates
    # df2 = pd.DataFrame(list(itertools.product(uuid1, dates2)), columns=['id', 'date'])
    # df2['v'] = np.random.sample(df2.shape[0])
    # r = d6tjoin.top1.MergeTop1(df1, df2, ['date'], ['date'], ['id'], ['id']).merge()
    # # why cause error?
    # r = d6tjoin.top1.MergeTop1(df1.head(), df2, ['date'], ['date'], ['id'], ['id']).merge()

    df2 = pd.DataFrame(list(itertools.product(uuid1, dates2)), columns=['id', 'date'])
    df2['v'] = np.random.sample(df2.shape[0])
    df2['id'] = df1['id'].str[1:-1]

    result = d6tjoin.top1.MergeTop1(df1, df2, ['date', 'id'], ['date', 'id']).merge()
    result['merged']
    # o=d6tjoin.top1.MergeTop1(df1, df2, ['date', 'id'], ['date', 'id'])
    # o.cfg_exact_left_on
    result = d6tjoin.top1.MergeTop1(df1, df2, ['date', 'id'], ['date', 'id']).merge()

    d6tjoin.utils.PreJoin([df1, df2], ['id', 'date']).stats_prejoin(print_only=False)

    assert True


def fiddle_set():

    import pandas as pd
    import numpy as np
    import importlib
    import d6tjoin.top1

    import ciseau
    import scipy.spatial.distance

    df_db = pd.read_csv('~/database.csv',index_col=0)

    def diff_jaccard(a, b):
        # pad with empty str to make euqal length
        a = np.pad(a, (0, max(0, len(b) - len(a))), 'constant', constant_values=(0, 0))
        b = np.pad(b, (0, max(0, len(a) - len(b))), 'constant', constant_values=(0, 0))
        return scipy.spatial.distance.jaccard(a, b)

    def strsplit(t):
        return [s for s in [s.replace(" ", "") for s in ciseau.tokenize(t)] if s not in ['.', ',', '-', ';', '(', ')']]

    importlib.reload(d6tjoin.top1)
    j = d6tjoin.top1.MergeTop1Diff(df_db.head(),df_db,'description','description',fun_diff=diff_jaccard,topn=2,fun_preapply=strsplit,fun_postapply=lambda x: ' '.join(x))
    j.merge()['merged']


def test_multicore():
    nobs = 10
    f1 = Faker()
    f1.seed(0)
    uuid1 = [str(f1.uuid4()).split('-')[0] for _ in range(nobs)]

    df1 = pd.DataFrame(uuid1, columns=['id'])
    df1['val1'] = np.round(np.random.sample(df1.shape[0]), 3)

    # create mismatch
    df2 = df1.copy()
    df2['id'] = df1['id'].str[1:-1]
    df2['val2'] = np.round(np.random.sample(df2.shape[0]), 3)


    m = d6tjoin.top1.MergeTop1Diff(df1,df2,'id','id',fun_diff=jellyfish.levenshtein_distance)
    df_candidates = m._allpairs_candidates()

    idxSel = df_candidates['__matchtype__'] != 'exact'
    dfd2 = df_candidates.copy()
    dfd2.loc[idxSel,'__top1diff__'] = d6tjoin.top1._applyFunMulticore(df_candidates.loc[idxSel,'__top1left__'].values, df_candidates.loc[idxSel,'__top1right__'].values,jellyfish.levenshtein_distance)

    dfd1 = df_candidates.copy()
    dfd1.loc[idxSel, '__top1diff__'] = df_candidates[idxSel].apply(lambda x: jellyfish.levenshtein_distance(x['__top1left__'], x['__top1right__']), axis=1)
    assert dfd2.equals(dfd1)

    assert True

    '''
    multicore in caller class
    pass multicore on
    make ifelse multicore for every apply diff
    
    default yes?
    part of requirements
    
    update setup.py requirements
    
    
    '''


test_top1_gen_candidates()