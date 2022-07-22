from ._diversity_indices_helper.aindex import _alpha_index
from ._diversity_indices_helper.bindex import _beta_index


def alpha_index(df, index=None, q=None, keep=False):
    return _alpha_index(df, index, q, keep)


def beta_index(df, index='w'):
    return _beta_index(df, index)
