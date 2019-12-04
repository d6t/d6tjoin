import pandas as pd
import numpy as np
from collections import OrderedDict
import itertools
import warnings
import jellyfish

from d6tjoin.pre import Prejoin as BaseJoin


# ******************************************
# helpers
# ******************************************
def set_values(dfg, key):
    v = dfg[key].unique()
    return v[~pd.isnull(v)]


def apply_gen_candidates_group(dfg):
    return pd.DataFrame(list(itertools.product(dfg['__top1left__'].values[0],dfg['__top1right__'].values[0])),columns=['__top1left__','__top1right__'])


def apply_gen_candidates(set1, set2):
    df_candidates = list(itertools.product(set1, set2))
    df_candidates = pd.DataFrame(df_candidates,columns=['__top1left__','__top1right__'])

    return df_candidates


def diff_arithmetic(x,y):
    return abs(x - y)


def diff_edit(a,b):
    return jellyfish.levenshtein_distance(a,b)


def filter_group_minmax(dfg, col):
    """

    Returns all rows equal to min in col

    """
    return dfg[dfg[col] == dfg[col].min()]


def prep_match_df(dfg):
    dfg = dfg[['__top1left__', '__top1right__', '__top1diff__', '__match type__']]
    return dfg

# ******************************************
# fuzzy join
# ******************************************
class FuzzyJoinTop1(BaseJoin):

    def __init__(self, dfs, exact_keys=[], fuzzy_keys=[], exact_how='inner', fuzzy_how = {}, keys_bydf=False, init_merge=False):

        """

        Smart joiner for top 1 similarity joins. By setting fuzzy keys, it calculates similarity metrics for strings, numbers and dates to join on the closest matching entry.

        Args:
            dfs (list): list of dataframes
            exact_keys (list): list of join keys for exact joins. See notes for details
            fuzzy_keys (list): list of join keys for fuzzy joins. See notes for details
            exact_how (str): exact join mode same as `pd.merge(how='inner')`
            fuzzy_how (dict): specify fuzzy join options by merge level eg {0:{'top_limit':1}}
            keys_bydf (bool): if keys list is by dataframe (default) or join level. See notes for details

        Note:
            * specifying join keys:
                * if both dataframes have matching columns: `fuzzy_keys=['key1','key2']`
                * else: `fuzzy_keys=[['key1df1','key1df2'],['key2df1','key2df2']]`
                    * by default you provide keys by join level eg `[['key1df1','key1df2'],['key2df1','key2df2']]` instead you can also provide keys by dataframe `[['key1df1','key2df1'],['key1df2','key2df2']], keys_bydf=True`
            * fuzzy_how: controls join options by join level
                * dict keys are join level eg with `fuzzy_keys=[['key1df1','key1df2'],['key2df1','key2df2']]` you set `fuzzy_how={0:{'top_nrecords':5},0:{'top_nrecords':5}}`
                * options are:
                    * fun_diff: difference function or list of difference functions applied sequentially. Needs to be 0=similar and >0 dissimilar
                    * top_limit: maximum difference, keep only canidates with difference <= top_limit
                    * top_nrecords: keep only n top_nrecords, good for generating previews

        """

        # inputs dfs
        self._init_dfs(dfs)

        # check and save join keys
        if not exact_keys and not fuzzy_keys:
            raise ValueError("Must provide at least one of exact_keys or fuzzy_keys")

        self.keys_exact, self.keysdf_exact = self._prep_keys(exact_keys, keys_bydf)
        if self.keys_exact:
            self._check_keysdfs(self.keys_exact, self.keysdf_exact)

        self.keys_fuzzy, self.keysdf_fuzzy = self._prep_keys(fuzzy_keys, keys_bydf)
        if self.keys_fuzzy:
            self._check_keysdfs(self.keys_fuzzy, self.keysdf_fuzzy)

        # todo: no duplicate join keys passed

        if not isinstance(exact_how, (str,)):
            raise NotImplementedError('exact_how can only be applied globally for now')
        elif exact_how not in ('left','right','inner','outer'):
            raise ValueError("Invalid how parameter, check documentation for valid values")

        self.cfg_njoins_exact = len(self.keysdf_exact[0]) if self.keysdf_exact else 0
        self.cfg_njoins_fuzzy = len(self.keysdf_fuzzy[0]) if self.keysdf_fuzzy else 0

        if self.cfg_njoins_fuzzy>1:
        #     raise NotImplementedError('Currently supports only 1 fuzzy key')
            warnings.warn('Multi-key fuzzy joins are currently done globally for each key indivudally, not hierarchically for each unique fuzzy key value pair')

        self.exact_how = exact_how
        self.set_fuzzy_how_all(fuzzy_how)

        if init_merge:
            self.join()
        else:
            self.dfjoined = None

        self.table_fuzzy = {}


    def set_fuzzy_how(self, ilevel, fuzzy_how):
        self.fuzzy_how[ilevel] = fuzzy_how
        self._gen_fuzzy_how(ilevel)

    def set_fuzzy_how_all(self, fuzzy_how):
        if not isinstance(fuzzy_how, (dict,)):
            raise ValueError('fuzzy_how needs to be a dict')
        self.fuzzy_how = fuzzy_how
        self._gen_fuzzy_how_all()

    def _gen_fuzzy_how_all(self):

        for ilevel in range(self.cfg_njoins_fuzzy):
            self._gen_fuzzy_how(ilevel)

    def _gen_fuzzy_how(self, ilevel):

            # check if entry exists
            cfg_top1 = self.fuzzy_how.get(ilevel,{})

            keyleft = self.keys_fuzzy[ilevel][0]
            keyright = self.keys_fuzzy[ilevel][1]

            typeleft = self.dfs[0][keyleft].dtype
            typeright = self.dfs[1][keyright].dtype

            if 'type' not in cfg_top1:
                if typeleft == 'int64' or typeleft == 'float64' or typeleft == 'datetime64[ns]':
                    cfg_top1['type'] = 'number'
                elif typeleft == 'object' and type(self.dfs[0][keyleft].values[~self.dfs[0][keyleft].isnull()][0])==str:
                    cfg_top1['type'] = 'string'
                else:
                    raise ValueError('Unrecognized data type for top match, need to pass fun_diff in arguments')

            # make defaults if no settings provided
            if 'fun_diff' not in cfg_top1:

                if cfg_top1['type'] == 'number':
                    cfg_top1['fun_diff'] = pd.merge_asof
                elif cfg_top1['type'] == 'string':
                    cfg_top1['fun_diff'] = diff_edit
                else:
                    raise ValueError('Unrecognized data type for top match, need to pass fun_diff in arguments')
            else:
                is_valid = callable(cfg_top1['fun_diff']) or (type(cfg_top1['fun_diff']) == list and all([callable(f) for f in cfg_top1['fun_diff']]))
                if not is_valid:
                    raise ValueError("'fun_diff' needs to be a function or a list of functions")

            if not type(cfg_top1['fun_diff']) == list:
                cfg_top1['fun_diff'] = [cfg_top1['fun_diff']]

            if 'top_limit' not in cfg_top1:
                cfg_top1['top_limit'] = None

            if 'top_nrecords' not in cfg_top1:
                cfg_top1['top_nrecords'] = None

            cfg_top1['dir'] = 'left'

            # save config
            # check if entry exists
            self.fuzzy_how[ilevel] = cfg_top1

    def preview_fuzzy(self, ilevel, top_nrecords=5):
        if top_nrecords>0:
            return self._gen_match_top1(ilevel, top_nrecords)
        else:
            return self._gen_match_top1(ilevel)

    def _gen_match_top1_left_number(self, cfg_group_left, cfg_group_right, keyleft, keyright, top_nrecords):
        if len(cfg_group_left) > 0:

            # unique values
            if top_nrecords is None:
                # df_keys_left = pd.DataFrame(self.dfs[0].groupby(cfg_group_left)[keyleft].unique())
                df_keys_left = self.dfs[0].groupby(cfg_group_left)[keyleft].apply(lambda x: pd.Series(x.unique()))
                df_keys_left.index = df_keys_left.index.droplevel(1)
                df_keys_left = pd.DataFrame(df_keys_left)
            else:
                # df_keys_left = pd.DataFrame(self.dfs[0].groupby(cfg_group_left)[keyleft].unique()[:top_nrecords])
                df_keys_left = self.dfs[0].groupby(cfg_group_left)[keyleft].apply(lambda x: pd.Series(x.unique()[:top_nrecords]))
                df_keys_left.index = df_keys_left.index.droplevel(1)
                df_keys_left = pd.DataFrame(df_keys_left)
            df_keys_right = self.dfs[1].groupby(cfg_group_right)[keyright].apply(lambda x: pd.Series(x.unique()))
            df_keys_right.index = df_keys_right.index.droplevel(1)
            df_keys_right = pd.DataFrame(df_keys_right)
            # df_keys_right = pd.DataFrame(self.dfs[1].groupby(cfg_group_right)[keyright].unique())

            # sort
            df_keys_left = df_keys_left.sort_values(keyleft).reset_index().rename(columns={keyleft:'__top1left__'})
            df_keys_right = df_keys_right.sort_values(keyright).reset_index().rename(columns={keyright:'__top1right__'})

            df_match = pd.merge_asof(df_keys_left, df_keys_right, left_on='__top1left__', right_on='__top1right__', left_by=cfg_group_left, right_by=cfg_group_right, direction='nearest')
        else:
            # uniques
            values_left = set_values(self.dfs[0], keyleft)
            values_right = set_values(self.dfs[1], keyright)

            if top_nrecords:
                values_left = values_left[:top_nrecords]

            df_keys_left = pd.DataFrame({'__top1left__':values_left}).sort_values('__top1left__')
            df_keys_right = pd.DataFrame({'__top1right__':values_right}).sort_values('__top1right__')

            df_match = pd.merge_asof(df_keys_left, df_keys_right, left_on='__top1left__', right_on='__top1right__', direction='nearest')

        df_match['__top1diff__'] = (df_match['__top1left__']-df_match['__top1right__']).abs()

        return df_match

    def _gen_match_top1(self, ilevel, top_nrecords=None):
        """

        Generates match table between two sets

        Args:
            keyssets (dict): values for join keys ['key left', 'key right', 'keyset left', 'keyset right', 'inner', 'outer', 'unmatched total', 'unmatched left', 'unmatched right']
            mode (str, list): global string or list for each join. Possible values: ['exact inner', 'exact left', 'exact right', 'exact outer', 'top1 left', 'top1 right', 'top1 bidir all', 'top1 bidir unmatched']
            is_lower_better (bool): True = difference, False = Similarity

        """

        cfg_top1 = self.fuzzy_how[ilevel]
        fun_diff = cfg_top1['fun_diff']
        top_limit = cfg_top1['top_limit']
        if not top_nrecords:
            top_nrecords = cfg_top1['top_nrecords']

        keyleft = self.keys_fuzzy[ilevel][0]
        keyright = self.keys_fuzzy[ilevel][1]

        #******************************************
        # table LEFT
        #******************************************
        if cfg_top1['dir']=='left':
            
            # exact keys for groupby
            cfg_group_left = self.keysdf_exact[0] if self.keysdf_exact else []
            cfg_group_right = self.keysdf_exact[1] if self.keysdf_exact else []

            if cfg_top1['type'] == 'string' or (cfg_top1['type'] == 'number' and cfg_top1['fun_diff'] != [pd.merge_asof]):

                if len(cfg_group_left)>0:
                    # generate candidates if exact matches are present (= blocking index)

                    if top_nrecords is None:
                        df_keys_left = pd.DataFrame(self.dfs[0].groupby(cfg_group_left)[keyleft].unique())
                    else:
                        df_keys_left = pd.DataFrame(self.dfs[0].groupby(cfg_group_left)[keyleft].unique()[:top_nrecords])
                    df_keys_right = pd.DataFrame(self.dfs[1].groupby(cfg_group_right)[keyright].unique())
                    df_keysets_groups = df_keys_left.merge(df_keys_right,left_index=True, right_index=True)
                    df_keysets_groups.columns = ['__top1left__','__top1right__']
                    dfg = df_keysets_groups.reset_index().groupby(cfg_group_left).apply(apply_gen_candidates_group)
                    dfg = dfg.reset_index(-1,drop=True).reset_index()
                    dfg = dfg.dropna()

                else:
                    # generate candidates if NO exact matches
                    values_left = set_values(self.dfs[0],keyleft)
                    values_right = set_values(self.dfs[1],keyright)

                    if top_nrecords is None:
                        dfg = apply_gen_candidates(values_left,values_right)
                    else:
                        dfg = apply_gen_candidates(values_left[:top_nrecords], values_right)


                # find exact matches and remove from candidates
                # todo: use set logic before generating candidates
                idxSelExact = dfg['__top1left__']==dfg['__top1right__']
                df_match_exact = dfg[idxSelExact].copy()
                df_match_exact['__match type__'] = 'exact'
                df_match_exact['__top1diff__'] = 0

                idxSel = dfg['__top1left__'].isin(df_match_exact['__top1left__'])
                dfg = dfg[~idxSel]

                for fun_diff in cfg_top1['fun_diff']:
                    dfg['__top1diff__'] = dfg.apply(lambda x: fun_diff(x['__top1left__'], x['__top1right__']), axis=1)

                    # filtering
                    if not top_limit is None:
                        dfg = dfg[dfg['__top1diff__'] <= top_limit]

                    # get top 1
                    dfg = dfg.groupby('__top1left__',group_keys=False).apply(lambda x: filter_group_minmax(x,'__top1diff__'))

                # return results
                dfg['__match type__'] = 'top1 left'
                df_match = pd.concat([dfg,df_match_exact])

            elif cfg_top1['type'] == 'number' and cfg_top1['fun_diff'] == [pd.merge_asof]:
                df_match = self._gen_match_top1_left_number(cfg_group_left, cfg_group_right, keyleft, keyright, top_nrecords).copy()

                # filtering
                if not top_limit is None:
                    df_match = df_match[df_match['__top1diff__'] <= top_limit]

                df_match['__match type__'] = 'top1 left'
                df_match.loc[df_match['__top1left__'] == df_match['__top1right__'], '__match type__'] = 'exact'
            else:
                raise ValueError('Dev error: cfg_top1["type/fun_diff"]')


        #******************************************
        # table RIGHT
        #******************************************
        elif cfg_top1['dir']=='right' or cfg_top1['dir'] == 'inner':
            raise NotImplementedError('Only use left join for now')
        else:
            raise ValueError("wrong 'how' parameter for top1 join, check documentation")

        return {'key left':keyleft, 'key right':keyright,
                'table':df_match,'has duplicates':df_match.groupby('__top1left__').size().max()>1}

    def run_match_top1_all(self, cfg_top1=None):

        for ilevel in range(self.cfg_njoins_fuzzy):
            self.table_fuzzy[ilevel] = self._gen_match_top1(ilevel)

    def join(self, is_keep_debug=False):
        if self.cfg_njoins_fuzzy==0:
            self.dfjoined = self.dfs[0].merge(self.dfs[1], left_on=self.keysdf_exact[0], right_on=self.keysdf_exact[1], how=self.exact_how)
        else:

            self.run_match_top1_all()

            cfg_group_left = self.keysdf_exact[0] if self.keysdf_exact else []
            cfg_group_right = self.keysdf_exact[1] if self.keysdf_exact else []
            self.dfjoined = self.dfs[0]
            for ilevel in range(self.cfg_njoins_fuzzy):
                keyleft = self.keys_fuzzy[ilevel][0]
                keyright = self.keys_fuzzy[ilevel][1]
                dft = self.table_fuzzy[ilevel]['table'].copy()
                dft.columns = [s + keyleft if s.startswith('__') else s for s in dft.columns]
                self.dfjoined = self.dfjoined.merge(dft, left_on=cfg_group_left+[keyleft], right_on=cfg_group_left+['__top1left__'+keyleft])
                pass

            cfg_keys_left = cfg_group_left+['__top1right__'+k for k in self.keysdf_fuzzy[0]]
            cfg_keys_right = cfg_group_right+[k for k in self.keysdf_fuzzy[1]]

            self.dfjoined = self.dfjoined.merge(self.dfs[1], left_on = cfg_keys_left, right_on = cfg_keys_right, suffixes=['','__right__'])

            if not is_keep_debug:
                self.dfjoined = self.dfjoined[self.dfjoined.columns[~self.dfjoined.columns.str.startswith('__')]]

        return self.dfjoined


