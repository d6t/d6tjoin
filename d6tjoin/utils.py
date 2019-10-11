from collections import OrderedDict

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import numpy as np

# ******************************************
# helpers
# ******************************************
def _set_values_series(dfs):
    return set(dfs[~pd.isnull(dfs)])

def _set_values(dfg, key):
    return _set_values_series(dfg[key])

def _filter_group_min(dfg, col, topn=1):
    """

    Returns all rows equal to min in col

    """
    if topn==1:
        return dfg[dfg[col] == dfg[col].min()]
    else:
        return dfg[dfg[col].isin(np.sort(dfg[col].unique())[:topn])]

from joblib import Parallel, delayed
import multiprocessing
def _applyFunMulticore(values1, values2, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(p[0],p[1]) for p in zip(values1,values2))
    return retLst


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

