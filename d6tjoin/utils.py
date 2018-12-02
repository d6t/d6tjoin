from collections import OrderedDict

import pandas as pd
import numpy as np


# ******************************************
# df_str_summary
# ******************************************

def _apply_strlen(dfg, unique_count=False):
    lenv = np.vectorize(len)
    alens = lenv(dfg.values)
    r = {'median':np.median(alens),'mean':np.mean(alens),'min':np.min(alens),'max':np.max(alens),'total':dfg.shape[0]}
    if unique_count:
        r['uniques'] = len(dfg.unique())
    return pd.Series(r)

def df_str_summary(dfg, columns=None, unique_count=False):
    """
    Returns statistics on length of all strings and other objects in pandas dataframe. Statistics include mean, median, min, max. Optional unique count.

    Args:
        dfg (dataframe): pandas dataframe
        columns (:obj:`list`, optional): column names to analyze. If None analyze all
        unique_count (:obj:`bool`, optional): include count of unique values

    Returns:
        dataframe: string length statistics
    """
    if not columns:
        columns=dfg.columns
    if unique_count:
        cfg_col_sel = ['mean','median','min','max','total','uniques']
    else:
        cfg_col_sel = ['mean','median','min','max','total']
    return dfg[columns].select_dtypes(include=['object']).apply(lambda x: _apply_strlen(x, unique_count)).T[cfg_col_sel]


# ******************************************
# tfidf
# ******************************************
import re
import collections
from joblib import Parallel, delayed
import multiprocessing
import itertools
import warnings

def tokenCount(dfs, fun, mincount=2, minlength=1):
    """
    Tokenize a series of strings and count occurance of string tokens

    Args:
        dfs (pd.series): pd.series of values
        fun (function): tokenize function
        mincount (int): discard tokens with count less than mincount
        minlength (int): discard tokens with string length less than minlength

    Returns:
        dataframe: count of tokens

    """
    assert len(dfs.shape)==1
    dfs=dfs.dropna().unique()
    
    if dfs.shape[0]>1000:
        words = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(fun)(s) for s in dfs)
    else:
        words = [fun(s) for s in dfs]
    words = list(itertools.chain.from_iterable(words))    
    df_count = [t for t in collections.Counter(words).most_common() if t[1]>=mincount and len(t[0])>=minlength]
    df_count = pd.DataFrame(df_count, columns=['word','count'])
    return df_count

def splitcharTokenCount(dfs, splitchars="[^a-zA-Z0-9]+", mincount=2, minlength=1): #"[ -_|]+"
    """
    Tokenize a series of strings by splitting strings on a set of characters. Then count occurance of tokens in series.

    Args:
        dfs (pd.series): pd.series of values
        splitchars (str): regex by which to split string into tokens. For example `"[^a-zA-Z0-9]+"` for anything not alpha-numeric or `"[ -_|]+"` for common ID tokens.
        mincount (int): discard tokens with count less than mincount
        minlength (int): discard tokens with string length less than minlength

    Returns:
        dataframe: count of tokens

    """
    def funsplit(s):
        return re.split(splitchars,s)
    return tokenCount(dfs, funsplit, mincount, minlength)

def ncharTokenCount(dfs, nchars=None, overlapping=False, mincount=2, minlength=1):
    """
    Tokenize a series of strings by splitting strings into tokens of `nchars` length. Then count occurance of tokens in series.

    Args:
        dfs (pd.series): pd.series of values
        nchars (int): number of characters in each token
        overlapping (bool): make overlapping tokens
        mincount (int): discard tokens with count less than mincount
        minlength (int): discard tokens with string length less than minlength

    Returns:
        dataframe: count of tokens

    """
    if not nchars:
        smax = dfs.str.len().max()
        smin = dfs.str.len().min()
        if smax-smin>2:
            warnings.warn('Tokenize works best if strings have similar length')
        nchars = dfs.str.len().max()//4

    if overlapping:
        def funtokenize(s):
            return [s[i:i+nchars] for i in range(0, len(s)-nchars+1)]
    else:
        def funtokenize(s):
            return [s[i:i+nchars] for i in range(0, len(s), nchars)]
    return tokenCount(dfs, funtokenize, mincount, minlength)


def unique_contains(dfs, strlist):
    """
    Find values which contain a set of substrings

    Args:
        dfs (pd.series): pd.series of values
        strlist (list): substrings to find

    Returns:
        list: unique values which contain substring

    """
    assert len(dfs.shape)==1
    dfs=np.unique(dfs)
    outlist = [(x, [s for s in dfs if x in s]) for x in strlist]
    return outlist


# ******************************************
# join base class
# ******************************************

class BaseJoin(object):

    def __init__(self, dfs, keys=None, keys_bydf=False):

        # inputs dfs
        self._init_dfs(dfs)

        # check and save join keys
        self._check_keys(keys)
        keys, keysdf = self._prep_keys(keys, keys_bydf)
        self._check_keysdfs(keys, keysdf)

        # todo: no duplicate join keys passed

        # join keys
        self.cfg_njoins = len(keysdf[0])
        self.keys = keys # keys by join level
        self.keysall = keys+[['__all__']*len(dfs)]
        self.keysdf = keysdf # keys by df
        self.keysdfall = keysdf+[['__all__']]*len(dfs)
        self.uniques = [] # set of unique values for each join key individually
        self.keysets = [] # set of unique values for all join keys together __all__

    def _init_dfs(self, dfs):
        # check and save dfs
        if len(dfs)<2:
            raise ValueError('Need to pass at least 2 dataframes')

        if len(dfs)>2:
            raise NotImplementedError('Only handles 2 dataframes for now')

        self.dfs = dfs
        self.cfg_ndfs = len(dfs)

    def _check_keys(self, keys):
        if not keys or len(keys)<1:
            raise ValueError("Need to have join keys")

    def _check_keysdfs(self, keys, keysdf):
        if not all([len(k)==len(self.dfs) for k in keys]):
            raise ValueError("Need to provide join keys for all dataframes")

        for idf,dfg in enumerate(self.dfs):
            dfg.head(1)[keysdf[idf]] # check that keys present in dataframe

    def _prep_keys(self, keys, keys_bydf):
        # deal with empty keys
        if not keys:
            return [], []

        # get keys in correct format given user input
        if isinstance(keys[0], (str,)):
            keysdf = [keys]*len(self.dfs)
            keys = list(map(list, zip(*keysdf)))

        elif isinstance(keys[0], (list,)):
            keysdf = list(map(list, zip(*keys)))

            if keys_bydf:
                keys, keysdf = keysdf, keys
                pass

        else:
            raise ValueError("keys need to be either list of strings or list of lists")

        return keys, keysdf


# ******************************************
# prejoin stats class
# ******************************************

class PreJoin(BaseJoin):
    """
    Analyze, slice & dice join keys and dataframes before joining. Useful for checking how good a join will be and quickly looking at unmatched join keys.

    Args:
        dfs (list): list of data frames to join
        keys (var): either list of strings `['a','b']` if join keys have the same names in all dataframes or list of lists if join keys are different across dataframes `[['a1','b1'],['a2','b2']]`
    """

    def _calc_keysets(self):

        self.keysets = [] # reset

        # find set of unique values for each join key
        for idx, dfg in enumerate(self.dfs):

            # keys individually
            uniquedict = OrderedDict()
            for key in self.keysdf[idx]:
                v = dfg[key].unique()
                uniquedict[key] = set(v[~pd.isnull(v)])

            # keys _all__
            dft = dfg[self.keysdf[idx]].drop_duplicates()
            uniquedict['__all__'] = {tuple(x) for x in dft.values}
            self.uniques.append(uniquedict)

        # perform set logic
        for keys in self.keysall:
            df_key = {}
            df_key['key left'] = keys[0]
            df_key['key right'] = keys[1]
            df_key['keyset left'] = self.uniques[0][df_key['key left']]
            df_key['keyset right'] = self.uniques[1][df_key['key right']]

            df_key['inner'] = df_key['keyset left'].intersection(df_key['keyset right'])
            df_key['outer'] = df_key['keyset left'].union(df_key['keyset right'])
            df_key['unmatched total'] = df_key['keyset left'].symmetric_difference(df_key['keyset right'])
            df_key['unmatched left'] = df_key['keyset left'].difference(df_key['keyset right'])
            df_key['unmatched right'] = df_key['keyset right'].difference(df_key['keyset left'])

            # check types are consistent
            vl = next(iter(df_key['keyset left'])) # take first element
            vr = next(iter(df_key['keyset right'])) # take first element

            df_key['value type'] = type(vl)

            self.keysets.append(df_key)


    def stats_prejoin(self, print_only=True, rerun=False):
        """
        Show prejoin statistics

        Args:
            return_results (bool): Return results as df instead of printing

        """

        if not self.keysets or rerun:
            self._calc_keysets()

        df_out = []

        for key_set in self.keysets:
            df_key = {}
            for k in ['keyset left','keyset right','inner','outer','unmatched total','unmatched left','unmatched right']:
                df_key[k] = len(key_set[k])
            for k in ['key left','key right']:
                df_key[k] = key_set[k]
            df_key['all matched'] = df_key['inner']==df_key['outer']
            df_out.append(df_key)

        df_out = pd.DataFrame(df_out)
        df_out = df_out.rename(columns={'keyset left':'left','keyset right':'right'})
        df_out = df_out[['key left','key right','all matched','inner','left','right','outer','unmatched total','unmatched left','unmatched right']]


        if print_only:
            print(df_out)
        else:
            return df_out

    def is_all_matched(self, key='__all__',rerun=False):

        if not self.keysets or rerun:
            self._calc_keysets()

        keymask = [key in e for e in self.keysall]
        if not (any(keymask)):
            raise ValueError('key ', self.cfg_show_key, ' not a join key in ', self.keys)
        ilevel = keymask.index(True)

        return (self.keysets[ilevel]['key left']==key or self.keysets[ilevel]['key right']==key) and len(self.keysets[ilevel]['unmatched total'])==0

    def show_input(self, nrows=3, keys_only=True, print_only=False):
        """
        .head() of input dataframes

        Args:
            keys_only (bool): only print join keys
            nrows (int): number of rows to show
            print (bool): print or return df

        """

        dfh = []
        for idf,dfg in enumerate(self.dfs):

            df = dfg

            if keys_only:
                df = dfg[self.keysdf[idf]]

            if nrows>0:
                df = df.head(nrows)

            if print_only:
                print('df #', idf)
                print(df)
            else:
                dfh.append(df)

        if not print_only:
            return dfh

    def _show_prep_df(self, idf, mode):
        """
        PRIVATE. prepare data for self.show() functions

        Args:
            idf (int): which df in self.dfs
            mode (str): matched vs unmatched

        """

        if idf==0:
            side='left'
        elif idf==1:
            side='right'
        else:
            raise ValueError('invalid idx')

        if self.cfg_show_keys_only:
            if self.cfg_show_key == '__all__':
                cfg_col_sel = self.keysdf[idf]
            else:
                cfg_col_sel = self.cfg_show_key
        else:
            cfg_col_sel = self.dfs[idf].columns

        # which set to return?
        if mode=='matched':
            cfg_mode_sel = 'inner'
        elif mode=='unmatched':
            cfg_mode_sel = mode + ' ' + side
        else:
            raise ValueError('invalid mode', mode)

        keys = list(self.keysets[self.cfg_show_level][cfg_mode_sel])
        if self.cfg_show_nrecords > 0:
            keys = keys[:self.cfg_show_nrecords]

        if self.cfg_show_key == '__all__' and self.cfg_njoins>1:
            dfg = self.dfs[idf].copy()
            dfg = self.dfs[idf].reset_index().set_index(self.keysdf[idf])
            dfg = dfg.loc[keys]
            dfg = dfg.reset_index().sort_values('index')[cfg_col_sel].reset_index(drop=True) # reorder to original order
        elif self.cfg_show_key == '__all__' and self.cfg_njoins==1:
            dfg = self.dfs[idf]
            dfg = dfg.loc[dfg[self.keysdf[idf][0]].isin([e[0] for e in keys]), cfg_col_sel]
        else:
            dfg = self.dfs[idf]
            dfg = dfg.loc[dfg[self.cfg_show_key].isin(keys),cfg_col_sel]

        if self.cfg_show_nrows > 0:
            dfg = dfg.head(self.cfg_show_nrows)

        if self.cfg_show_print_only:
            print('%s %s for key %s' %(mode, side, self.cfg_show_key))
            print(dfg)
        else:
            self.df_show_out[side] = dfg.copy()

    def _show(self, mode):
        if not self.keysets:
            raise RuntimeError('run .stats_prejoin() first')

        keymask = [self.cfg_show_key in e for e in self.keysall]
        if not (any(keymask)):
            raise ValueError('key ', self.cfg_show_key, ' not a join key in ', self.keys)
        self.cfg_show_level = keymask.index(True)

        for idf in range(self.cfg_ndfs):  # run for all self.dfs
            if self.keysall[self.cfg_show_level][idf] == self.cfg_show_key:  # check if key applies
                self._show_prep_df(idf, mode)

    def show_unmatched(self, key, nrecords=3, nrows=3, keys_only=False, print_only=False):
        """
        Show unmatched records

        Args:
            key (str): join key
            nrecords (int): number of unmatched records
            nrows (int): number of rows
            keys_only (bool): show only join keys
            print_only (bool): if false return results instead of printing
        """
        self.df_show_out = {}
        self.cfg_show_key = key
        self.cfg_show_nrecords = nrecords
        self.cfg_show_nrows = nrows
        self.cfg_show_keys_only = keys_only
        self.cfg_show_print_only = print_only

        self._show('unmatched')
        if not self.cfg_show_print_only:
            return self.df_show_out

    def show_matched(self, key, nrecords=3, nrows=3, keys_only=False, print_only=False):
        """
        Show matched records

        Args:
            key (str): join key
            nrecords (int): number of unmatched records
            nrows (int): number of rows
            keys_only (bool): show only join keys
            print_only (bool): if false return results instead of printing
        """
        self.df_show_out = {}
        self.cfg_show_key = key
        self.cfg_show_nrecords = nrecords
        self.cfg_show_nrows = nrows
        self.cfg_show_keys_only = keys_only
        self.cfg_show_print_only = print_only

        self._show('matched')
        if not self.cfg_show_print_only:
            return self.df_show_out
