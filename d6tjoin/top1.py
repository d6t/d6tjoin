import pandas as pd
import numpy as np
from collections import OrderedDict
import itertools
import warnings
import jellyfish

# ******************************************
# helpers
# ******************************************


from d6tjoin.utils import _applyFunMulticore, _filter_group_min, _set_values

class MergeTop1Diff(object):
    """

    Top1 minimum difference join. Mostly used for strings. Helper for `MergeTop1`.

    """

    def __init__(self, df1, df2, fuzzy_left_on, fuzzy_right_on, fun_diff=None, exact_left_on=None, exact_right_on=None,
                 top_limit=None, topn=1, fun_preapply = None, fun_postapply = None, is_keep_debug=False, use_multicore=True):

        # check exact keys
        if not exact_left_on:
            exact_left_on = []
        if not exact_right_on:
            exact_right_on = []

        if not isinstance(fuzzy_left_on, (str,)) or not isinstance(fuzzy_right_on, (str,)):
            raise ValueError('fuzzy_on needs to be a string')

        if len(exact_left_on) != len(exact_right_on):
            raise ValueError('Need to pass same number of exact keys')
        if not isinstance(exact_left_on, (list)) or not isinstance(exact_right_on, (list)):
            raise ValueError('Exact keys need to be a list')

        if not callable(fun_diff):
            raise ValueError('fun_diff needs to a function')

        if (fun_preapply and fun_postapply) and (not callable(fun_preapply) or not callable(fun_postapply)):
            raise ValueError('fun_preapply and fun_postapply needs to a function')

        # use blocking index?
        if not exact_left_on and not exact_right_on:
            self.cfg_is_block = False
        elif exact_left_on and exact_right_on:
            self.cfg_is_block = True
        else:
            raise ValueError('Need to pass exact keys for both or neither dataframe')

        # store data
        self.dfs = [df1,df2]

        # store config
        self.cfg_fuzzy_left_on = fuzzy_left_on
        self.cfg_fuzzy_right_on = fuzzy_right_on
        self.cfg_exact_left_on = exact_left_on
        self.cfg_exact_right_on = exact_right_on
        self.cfg_fun_diff = fun_diff
        self.cfg_fun_preapply = fun_preapply
        self.cfg_fun_postapply = fun_postapply
        self.cfg_top_limit = top_limit
        self.cfg_is_keep_debug = is_keep_debug
        self.cfg_topn = topn
        self.cfg_use_multicore = use_multicore

    def _allpairs_candidates(self):
        values_left = _set_values(self.dfs[0], self.cfg_fuzzy_left_on)
        values_right = _set_values(self.dfs[1], self.cfg_fuzzy_right_on)

        if self.cfg_topn>1:
            values_left_exact = set()
            values_left_fuzzy = values_left
        else:
            values_left_exact = values_left.intersection(values_right)
            values_left_fuzzy = values_left.difference(values_right)

        # pre apply a function
        if self.cfg_fun_preapply:
            values_left_fuzzy = [self.cfg_fun_preapply(v) for v in values_left_fuzzy]
            values_right = [self.cfg_fun_preapply(v) for v in values_right]

        df_candidates_fuzzy = list(itertools.product(values_left_fuzzy, values_right))
        df_candidates_fuzzy = pd.DataFrame(df_candidates_fuzzy,columns=['__top1left__','__top1right__'])
        df_candidates_fuzzy['__matchtype__'] = 'top1 left'

        df_candidates_exact = pd.DataFrame({'__top1left__': list(values_left_exact)})
        df_candidates_exact['__top1right__'] = df_candidates_exact['__top1left__']
        df_candidates_exact['__matchtype__'] = 'exact'

        df_candidates = df_candidates_exact.append(df_candidates_fuzzy, ignore_index=True)

        return df_candidates

    def _top1_diff_noblock(self):
        df_candidates = self._allpairs_candidates()

        idxSel = df_candidates['__matchtype__'] != 'exact'
        if self.cfg_use_multicore:
            df_candidates.loc[idxSel, '__top1diff__'] = _applyFunMulticore(df_candidates.loc[idxSel,'__top1left__'].values, df_candidates.loc[idxSel,'__top1right__'].values,self.cfg_fun_diff)
        else:
            df_candidates.loc[idxSel,'__top1diff__'] = df_candidates[idxSel].apply(lambda x: self.cfg_fun_diff(x['__top1left__'], x['__top1right__']), axis=1)

        df_candidates.loc[~idxSel, '__top1diff__'] = 0
        has_duplicates = False

        if self.cfg_fun_postapply:
            df_candidates['__top1left__']=df_candidates['__top1left__'].apply(self.cfg_fun_postapply,1)
            df_candidates['__top1right__']=df_candidates['__top1right__'].apply(self.cfg_fun_postapply,1)

        df_diff = df_candidates.groupby('__top1left__',group_keys=False).apply(lambda x: _filter_group_min(x,'__top1diff__',self.cfg_topn))
        if self.cfg_top_limit is not None:
            df_diff = df_diff[df_diff['__top1diff__']<=self.cfg_top_limit]
        has_duplicates = df_diff.groupby('__top1left__').size().max()>1
        if has_duplicates:
            warnings.warn('Top1 join for %s has duplicates' %self.cfg_fuzzy_left_on)

        return df_diff, has_duplicates


    def _merge_top1_diff_noblock(self):
        df_diff, has_duplicates = self._top1_diff_noblock()
        dfjoin = self.dfs[0].merge(df_diff, left_on=self.cfg_fuzzy_left_on, right_on='__top1left__')
        dfjoin = dfjoin.merge(self.dfs[1], left_on='__top1right__', right_on=self.cfg_fuzzy_right_on, suffixes=['','__right__'])

        if not self.cfg_is_keep_debug:
            dfjoin = dfjoin[dfjoin.columns[~dfjoin.columns.str.startswith('__')]]

        return {'merged':dfjoin, 'top1':df_diff, 'duplicates':has_duplicates}


    def _top1_diff_withblock(self):

        def apply_gen_candidates_group(dfg):
            return pd.DataFrame(list(itertools.product(dfg['__top1left__'].values[0],dfg['__top1right__'].values[0])),columns=['__top1left__','__top1right__'])

        # find key unique values
        keysleft = self.dfs[0][self.cfg_exact_left_on+[self.cfg_fuzzy_left_on]].drop_duplicates().dropna()
        keysright = self.dfs[1][self.cfg_exact_right_on+[self.cfg_fuzzy_right_on]].drop_duplicates().dropna()
        keysleft = {tuple(x) for x in keysleft.values}
        keysright = {tuple(x) for x in keysright.values}
        values_left_exact = keysleft.intersection(keysright)
        values_left_fuzzy = keysleft.difference(keysright)

        df_keys_left_exact = pd.DataFrame(list(values_left_exact))
        if not df_keys_left_exact.empty:
            df_keys_left_exact.columns = self.cfg_exact_left_on+['__top1left__']
            df_keys_left_exact['__top1right__']=df_keys_left_exact['__top1left__']
            df_keys_left_exact['__matchtype__'] = 'exact'

        df_keys_left_fuzzy = pd.DataFrame(list(values_left_fuzzy))
        if not df_keys_left_fuzzy.empty:
            df_keys_left_fuzzy.columns = self.cfg_exact_left_on+[self.cfg_fuzzy_left_on]

        # fuzzy pair candidates
        df_keys_left = pd.DataFrame(df_keys_left_fuzzy.groupby(self.cfg_exact_left_on)[self.cfg_fuzzy_left_on].unique())
        df_keys_right = pd.DataFrame(self.dfs[1].groupby(self.cfg_exact_right_on)[self.cfg_fuzzy_right_on].unique())
        df_keysets_groups = df_keys_left.merge(df_keys_right, left_index=True, right_index=True)
        df_keysets_groups.columns = ['__top1left__', '__top1right__']
        df_keysets_groups = df_keysets_groups.reset_index().groupby(self.cfg_exact_left_on).apply(apply_gen_candidates_group)
        df_keysets_groups = df_keysets_groups.reset_index(-1, drop=True).reset_index()
        df_keysets_groups = df_keysets_groups.dropna()

        df_candidates = df_keysets_groups[['__top1left__', '__top1right__']].drop_duplicates()
        if self.cfg_use_multicore:
            df_candidates['__top1diff__'] = _applyFunMulticore(df_candidates['__top1left__'].values, df_candidates['__top1right__'].values, self.cfg_fun_diff)
        else:
            df_candidates['__top1diff__'] = df_candidates.apply(lambda x: self.cfg_fun_diff(x['__top1left__'], x['__top1right__']), axis=1)
        df_candidates['__matchtype__'] = 'top1 left'

        # calculate difference
        df_diff = df_keysets_groups.merge(df_candidates, on=['__top1left__', '__top1right__'])

        df_diff = df_diff.append(df_keys_left_exact)
        df_diff['__top1diff__']=df_diff['__top1diff__'].fillna(0) # exact keys
        df_diff = df_diff.groupby(self.cfg_exact_left_on+['__top1left__'],group_keys=False).apply(lambda x: _filter_group_min(x,'__top1diff__'))
        if self.cfg_top_limit is not None:
            df_diff = df_diff[df_diff['__top1diff__']<=self.cfg_top_limit]
        has_duplicates = df_diff.groupby(self.cfg_exact_left_on+['__top1left__']).size().max()>1

        return df_diff, has_duplicates


    def _merge_top1_diff_withblock(self):

        df_diff, has_duplicates = self._top1_diff_withblock()

        dfjoin = self.dfs[0].merge(df_diff, left_on=self.cfg_exact_left_on+[self.cfg_fuzzy_left_on], right_on=self.cfg_exact_left_on+['__top1left__'])
        # todo: add exact join keys
        dfjoin = dfjoin.merge(self.dfs[1], left_on=self.cfg_exact_left_on+['__top1right__'], right_on=self.cfg_exact_right_on+[self.cfg_fuzzy_right_on], suffixes=['','__right__'])

        if not self.cfg_is_keep_debug:
            dfjoin = dfjoin[dfjoin.columns[~dfjoin.columns.str.startswith('__')]]

        return {'merged':dfjoin, 'top1':df_diff, 'duplicates':has_duplicates}

    def top1_diff(self):
        if self.cfg_is_block:
            return self._top1_diff_withblock()
        else:
            return self._top1_diff_noblock()

    def merge(self):

        if not self.cfg_exact_left_on and not self.cfg_exact_right_on:
            return self._merge_top1_diff_noblock()
        elif self.cfg_exact_left_on and self.cfg_exact_right_on:
            return self._merge_top1_diff_withblock()
        else:
            raise ValueError('Need to pass exact keys for both or neither dataframe')


class MergeTop1Number(object):
    """

    Top1 minimum difference join for numbers. Helper for `MergeTop1`.

    """

    def __init__(self, df1, df2, fuzzy_left_on, fuzzy_right_on, exact_left_on=None, exact_right_on=None,
                 direction='nearest', top_limit=None, is_keep_debug=False):

        # check exact keys
        if not exact_left_on:
            exact_left_on = []
        if not exact_right_on:
            exact_right_on = []

        if len(exact_left_on) != len(exact_right_on):
            raise ValueError('Need to pass same number of exact keys')
        if not isinstance(exact_left_on, (list)) or not isinstance(exact_right_on, (list)):
            raise ValueError('Exact keys need to be a list')

        # use blocking index?
        if not exact_left_on and not exact_right_on:
            self.cfg_is_block = False
        elif exact_left_on and exact_right_on:
            self.cfg_is_block = True
        else:
            raise ValueError('Need to pass exact keys for both or neither dataframe')

        # store data
        self.dfs = [df1,df2]

        # store config
        self.cfg_fuzzy_left_on = fuzzy_left_on
        self.cfg_fuzzy_right_on = fuzzy_right_on
        self.cfg_exact_left_on = exact_left_on
        self.cfg_exact_right_on = exact_right_on
        self.cfg_direction = direction
        self.cfg_top_limit = top_limit
        self.cfg_is_keep_debug = is_keep_debug

    def _top1_diff_withblock(self):

        # unique values
        df_keys_left = self.dfs[0].groupby(self.cfg_exact_left_on)[self.cfg_fuzzy_left_on].apply(lambda x: pd.Series(x.unique()))
        df_keys_left.index = df_keys_left.index.droplevel(-1)
        df_keys_left = pd.DataFrame(df_keys_left)
        df_keys_right = self.dfs[1].groupby(self.cfg_exact_right_on)[self.cfg_fuzzy_right_on].apply(lambda x: pd.Series(x.unique()))
        df_keys_right.index = df_keys_right.index.droplevel(-1)
        df_keys_right = pd.DataFrame(df_keys_right)

        # todo: global consolidation like with MergeTop1Diff

        # sort
        df_keys_left = df_keys_left.sort_values(self.cfg_fuzzy_left_on).reset_index().rename(columns={self.cfg_fuzzy_left_on:'__top1left__'})
        df_keys_right = df_keys_right.sort_values(self.cfg_fuzzy_right_on).reset_index().rename(columns={self.cfg_fuzzy_right_on:'__top1right__'})

        # merge
        df_diff = pd.merge_asof(df_keys_left, df_keys_right, left_on='__top1left__', right_on='__top1right__', left_by=self.cfg_exact_left_on, right_by=self.cfg_exact_right_on, direction=self.cfg_direction)
        df_diff['__top1diff__'] = (df_diff['__top1left__']-df_diff['__top1right__']).abs()
        df_diff['__matchtype__'] = 'top1 left'
        df_diff.loc[df_diff['__top1left__'] == df_diff['__top1right__'], '__matchtype__'] = 'exact'
        if self.cfg_top_limit is not None:
            df_diff = df_diff[df_diff['__top1diff__']<=self.cfg_top_limit]

        return df_diff

    def _top1_diff_noblock(self):
            # uniques
            values_left = _set_values(self.dfs[0], self.cfg_fuzzy_left_on)
            values_right = _set_values(self.dfs[1], self.cfg_fuzzy_right_on)

            # sort
            df_keys_left = pd.DataFrame({'__top1left__':list(values_left)}).sort_values('__top1left__')
            df_keys_right = pd.DataFrame({'__top1right__':list(values_right)}).sort_values('__top1right__')

            # merge
            df_diff = pd.merge_asof(df_keys_left, df_keys_right, left_on='__top1left__', right_on='__top1right__', direction=self.cfg_direction)
            df_diff['__top1diff__'] = (df_diff['__top1left__']-df_diff['__top1right__']).abs()
            df_diff['__matchtype__'] = 'top1 left'
            df_diff.loc[df_diff['__top1left__'] == df_diff['__top1right__'], '__matchtype__'] = 'exact'

            return df_diff

    def top1_diff(self):
        if self.cfg_is_block:
            return self._top1_diff_withblock()
        else:
            return self._top1_diff_noblock()

    def merge(self):
        df_diff = self.top1_diff()

        dfjoin = self.dfs[0].merge(df_diff, left_on=self.cfg_exact_left_on+[self.cfg_fuzzy_left_on], right_on=self.cfg_exact_left_on+['__top1left__'])
        dfjoin = dfjoin.merge(self.dfs[1], left_on=self.cfg_exact_left_on+['__top1right__'], right_on=self.cfg_exact_right_on+[self.cfg_fuzzy_right_on], suffixes=['','__right__'])

        if not self.cfg_is_keep_debug:
            dfjoin = dfjoin[dfjoin.columns[~dfjoin.columns.str.startswith('__')]]

        return {'merged': dfjoin, 'top1': df_diff, 'duplicates': None}

class MergeTop1(object):
    """

    Left best match join. It applies a difference function to find the key pair with the smallest difference to the join key.

    Args:
        df1 (dataframe): left dataframe onto which the right dataframe is joined
        df2 (dataframe): right dataframe
        fuzzy_left_on (list): join keys for similarity match, left dataframe
        fuzzy_right_on (list): join keys for similarity match, right dataframe
        exact_left_on (list, default None): join keys for exact match, left dataframe
        exact_right_on (list, default None): join keys for exact match, right dataframe
        fun_diff (list, default None): list of difference functions to be applied for each fuzzy key
        top_limit (list, default None): list of values to cap similarity matches
        is_keep_debug (bool): keep diagnostics columns, good for debugging

    Note:
        * fun_diff: applies the difference function to find the best match with minimum distance
            * By default gets automatically determined depending on whether you have a string or date/number
            * Use `None` to keep the default, so example [None, lambda x, y: x-y]
            * Functions within list get applied in order same order to fuzzy join keys
            * Needs to be a difference function so lower is better. For functions like Jaccard higher is better so you need to adjust for that
        * top_limit: Limits the number of matches to anything below that values. For example if two strings differ by 3 but top_limit is 2, that match will be ignored
            * for dates you can use `pd.offsets.Day(1)` or similar

    """

    def __init__(self, df1, df2, fuzzy_left_on=None, fuzzy_right_on=None, exact_left_on=None, exact_right_on=None,
                 fun_diff = None, top_limit=None, is_keep_debug=False, use_multicore=True):


        # todo: pass custom merge asof param
        # todo: pass list of fundiff


        # check fuzzy keys
        if not fuzzy_left_on or not fuzzy_right_on:
            raise ValueError('Need to pass fuzzy left and right keys')
        if len(fuzzy_left_on) != len(fuzzy_right_on):
            raise ValueError('Need to pass same number of fuzzy left and right keys')
        self.cfg_njoins_fuzzy = len(fuzzy_left_on)

        # check exact keys
        if not exact_left_on:
            exact_left_on = []
        if not exact_right_on:
            exact_right_on = []

        if len(exact_left_on) != len(exact_right_on):
            raise ValueError('Need to pass same number of exact keys')
        if not isinstance(exact_left_on, (list)) or not isinstance(exact_right_on, (list)):
            raise ValueError('Exact keys need to be a list')


        # use blocking index?
        if not exact_left_on and not exact_right_on:
            self.cfg_is_block = False
        elif exact_left_on and exact_right_on:
            self.cfg_is_block = True
        else:
            raise ValueError('Need to pass exact keys for both or neither dataframe')

        # check custom params
        if not top_limit:
            top_limit = [None,]*self.cfg_njoins_fuzzy
        if not fun_diff:
            fun_diff = [None,]*self.cfg_njoins_fuzzy
        elif len(fun_diff)!=len(fuzzy_left_on):
            raise ValueError('fun_diff needs to the same length as fuzzy_left_on. Use None in list to use default')
        if not isinstance(top_limit, (list,)) or not len(top_limit)==self.cfg_njoins_fuzzy:
            raise NotImplementedError('top_limit needs to a list with entries for each fuzzy join key')
        if not isinstance(fun_diff, (list,)) or not len(top_limit)==self.cfg_njoins_fuzzy:
            raise NotImplementedError('fun_diff needs to a list with entries for each fuzzy join key')

        # store data
        self.dfs = [df1,df2]

        # store config
        self.cfg_fuzzy_left_on = fuzzy_left_on
        self.cfg_fuzzy_right_on = fuzzy_right_on
        # todo: exact keys by fuzzy key? or just global?
        self.cfg_exact_left_on = exact_left_on
        self.cfg_exact_right_on = exact_right_on
        self.cfg_top_limit = top_limit
        self.cfg_fun_diff = fun_diff
        self.cfg_is_keep_debug = is_keep_debug
        self.cfg_use_multicore = use_multicore

    def merge(self):
        """

        Executes merge

        Returns:
             dict: keys 'merged' has merged dataframe, 'top1' has best matches by fuzzy_left_on. See example notebooks for details

        """
        df_diff_bylevel = OrderedDict()

        self.dfjoined = self.dfs[0].copy()
        cfg_exact_left_on = self.cfg_exact_left_on
        cfg_exact_right_on = self.cfg_exact_right_on

        a=1
        for ilevel, ikey in enumerate(self.cfg_fuzzy_left_on):
            keyleft = ikey
            keyright = self.cfg_fuzzy_right_on[ilevel]
            typeleft = self.dfs[0][keyleft].dtype

            if self.cfg_fun_diff[ilevel]:
                df_diff_bylevel[ikey] = MergeTop1Diff(self.dfjoined, self.dfs[1], keyleft, keyright, self.cfg_fun_diff[ilevel], cfg_exact_left_on, cfg_exact_right_on, top_limit=self.cfg_top_limit[ilevel], use_multicore=self.cfg_use_multicore).top1_diff()[0]
            else:
                if typeleft == 'int64' or typeleft == 'float64' or typeleft == 'datetime64[ns]':
                    df_diff_bylevel[ikey] = MergeTop1Number(self.dfjoined, self.dfs[1], keyleft, keyright, cfg_exact_left_on, cfg_exact_right_on, top_limit=self.cfg_top_limit[ilevel]).top1_diff()
                elif typeleft == 'object' and type(self.dfs[0][keyleft].values[0])==str:
                    df_diff_bylevel[ikey] = MergeTop1Diff(self.dfjoined, self.dfs[1], keyleft, keyright, jellyfish.levenshtein_distance, cfg_exact_left_on, cfg_exact_right_on, top_limit=self.cfg_top_limit[ilevel], use_multicore=self.cfg_use_multicore).top1_diff()[0]
                    # todo: handle duplicates
                else:
                    raise ValueError('Unrecognized data type for top match, need to pass fun_diff in arguments')

            self.dfjoined = self.dfjoined.merge(df_diff_bylevel[ikey], left_on=cfg_exact_left_on+[keyleft], right_on=cfg_exact_left_on+['__top1left__'], suffixes=['',keyleft])
            cfg_col_rename = ['__top1left__','__top1right__','__top1diff__','__matchtype__']
            self.dfjoined = self.dfjoined.rename(columns=dict((k,k+keyleft) for k in cfg_col_rename))
            cfg_exact_left_on += ['__top1right__%s'%keyleft,]
            cfg_exact_right_on += [keyright,]

        self.dfjoined = self.dfjoined.merge(self.dfs[1], left_on=cfg_exact_left_on, right_on=cfg_exact_right_on, suffixes=['','_right'])

        if not self.cfg_is_keep_debug:
            self.dfjoined = self.dfjoined[self.dfjoined.columns[~self.dfjoined.columns.str.startswith('__')]]

        return {'merged': self.dfjoined, 'top1': df_diff_bylevel, 'duplicates': None}

'''
multikey: want to merge left match onto right df
dont to numbers (non key) join until the very end
'''