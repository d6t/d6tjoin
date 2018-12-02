# Databolt Smart Join

Easily join different datasets without writing custom code. Does best match joins on strings, dates and numbers. For example you can quickly join similar but not identical stock tickers, addresses, names and dates without manual processing.

## Sample Use

```

import d6tjoin.top1
import d6tjoin.utils

#************************
# pre join diagnostics
#************************

# check join quality => none of the ids match

d6tjoin.utils.PreJoin([df1,df2],['id','date']).stats_prejoin()

  key left key right  all matched  inner  left  right  outer  unmatched total  unmatched left  unmatched right
0       id        id        False      0    10     10     20               20              10               10
1     date      date         True    366   366    366    366                0               0                0
2  __all__   __all__        False      0  3660   3660   7320             7320            3660             3660

#************************
# best match join on id
#************************

result = d6tjoin.top1.MergeTop1(df1,df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'],
    exact_left_on=['date'],exact_right_on=['date']).merge()

result['merged'].head(2)

        date        id   val1 id_right  val1_right   val2
0 2010-01-01  e3e70682  0.020   3e7068       0.020  0.034
1 2010-01-01  f728b4fa  0.806   728b4f       0.806  0.849

#************************
# debug best matches
#************************

result['top1']['id'].head(2)

         date __top1left__ __top1right__  __top1diff__ __matchtype__
10 2010-01-01     e3e70682        3e7068             2     top1 left
34 2010-01-01     e443df78        443df7             2     top1 left

#************************
# customize similarity fct
#************************
import affinegap

result = d6tjoin.top1.MergeTop1(df1,df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'], 
    fun_diff=[affinegap.affineGapDistance]).merge()

#************************
# token-based substring clusters and joins
#************************
dftoken=d6tjoin.utils.splitcharTokenCount(df2['id'])

      word  count
0   Equity      7
1       US      5
2       NA      2
3  PRIVATE      2


d6tjoin.utils.unique_contains(df2['id'], dftoken['word'].values)
>>> [('Equity', ['AAPL_US_Equity', 'AAP_US_Equity', 'AD_NA_Equity', 'AMZN_US_Equity', 'APRN_US_Equity', 'BBY_US_Equity', 'BMW_NA_Equity']), ('US', ['AAPL_US_Equity', 'AAP_US_Equity', 'AMZN_US_Equity', 'APRN_US_Equity', 'BBY_US_Equity']), ('NA', ['AD_NA_Equity', 'BMW_NA_Equity']), ('PRIVATE', ['PRIVATE_JLP', 'PRIVATE_NMG'])]

import re
splitchars="[^a-zA-Z0-9]+"
def tokenmatch(s1,s2):
    return 3-len(set(re.split(splitchars,s1)) & set(re.split(splitchars,s2)))

d6tjoin.top1.MergeTop1(df1,df2,fuzzy_left_on=['id'],fuzzy_right_on=['id'], fun_diff=[tokenmatch], top_limit=[2]).merge()['top1']['id']

  __top1left__   __top1right__ __matchtype__  __top1diff__
0          AAP   AAP_US_Equity     top1 left             2
1         AAPL  AAPL_US_Equity     top1 left             2
2    AMZN-AMZN  AMZN_US_Equity     top1 left             2
3         APRN  APRN_US_Equity     top1 left             2
4          JLP     PRIVATE_JLP     top1 left             2
5          NMG     PRIVATE_NMG     top1 left             2

```

## Features include
Enhances `pd.merge()` function with:
* Pre join diagnostics to identify mismatched join keys
* Best match joins that finds the top1 most similar value
	* Quickly join stock identifiers, addresses, names without manual processing
	* Ability to customize similarity functions, set max difference and other advanced features

## Installation

We recommend using the latest version from github `pip install git+https://github.com/d6t/d6tjoin.git`

If you cannot install from github, use the latest published version `pip install d6tjoin`.

We recommend using [AffineGap](https://github.com/dedupeio/affinegap) which is not an official requirement, you can install using `pip install affinegap`.

For the `jellyfish` library, make sure the C implementation is working else `d6tjoin` will be very slow. You can test by running `import jellyfish.cjellyfish` if the C version is installed. If you don't have a C compiler, you can `conda install -c conda-forge jellyfish`.

## Documentation

*  [PreJoin examples notebook](https://github.com/d6t/d6tjoin/blob/master/examples-prejoin.ipynb) - Examples for diagnosing join problems
*  [MergeTop1 notebook](https://github.com/d6t/d6tjoin/blob/master/examples-top1.ipynb) - Best match join examples notebook
*  [Token substring join notebook](https://github.com/d6t/d6tjoin/blob/master/examples-tokencluster.ipynb) - Find common substrings and joins on token substrings
*  [Official docs](http://d6tjoin.readthedocs.io/en/latest/py-modindex.html) - Detailed documentation for modules, classes, functions
