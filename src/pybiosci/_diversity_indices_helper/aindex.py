import numpy as np
import pandas as pd


def _get_index_value(df, index=None, q=None, keep=False):
    if index == "Hill":
        if q is None:
            raise ValueError("Please specify a q value")

        return hill_numbers(df, q, keep)

    elif index == "Renyi":
        if q is None:
            raise ValueError("Please specify a q value")

        return renyi_entropy(df, q, keep)

    elif index == "BergerParker":
        return berger_parker(df, keep)

    elif index == "Richness":
        return observed_richness(df)

    elif index == "iSimpson":
        return inverse_simpson(df)

    elif index == "gSimpson":
        return ginni_simpson(df)

    elif index == "Shannon":
        return shannon_entropy(df)

    elif index == "Chao1":
        return chao1(df)

    elif index == "ACE":
        return abundance_coverage_estimator(df)

    elif index == "Jackknife1":
        return first_order_jackknife(df)

    elif index == "Jackknife2":
        return second_order_jackknife(df)

    elif index == "Pielou":
        return Pielou(df, keep)

    elif index == "Tail":
        return tail(df)

    elif index == "EF":
        if q is None:
            raise ValueError("Please specify a q value")

        return evenness_factor(df, q, keep)

    elif index == "IF":
        if q is None:
            raise ValueError("Please specify a q value")

        return inverse_evenness_factor(df, q, keep)

    elif index == "RLE":
        if q is None:
            raise ValueError("Please specify a q value")

        return relative_evenness(df, q, keep)

    elif index == "RLI":
        if q is None:
            raise ValueError("Please specify a q value")

        return inverse_relative_evenness(df, q, keep)


def _alpha_index(df, index='shannon', q=None, keep=False):
    all_indices = ["Hill", "Renyi", "BergerParker",
                   "Richness", "iSimpson", "gSimpson", "Shannon",
                   "Chao1", "ACE", "Jackknife1", "Jackknife2",
                   "Pielou", "Tail", "EF", "IF", "RLE", "RLI"]

    if index != "All" and index not in all_indices:
        raise ValueError("Please specify one of the following indices\n" + ", ".join(all_indices))

    if index == "All":
        all_indices_df = pd.DataFrame()
        for idx in all_indices:
            all_indices_df[idx] = _get_index_value(df, index=idx, q=2)

        return all_indices_df

    else:
        return _get_index_value(df, index, q, keep)


def reformat_file(df):
    if not all(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).values):
        return False, "Please make sure all the values are numeric"

    df = df.replace(0, np.nan)
    return True, df


def qd(in_data, q, keep=False):
    D = None
    if not all(in_data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).values):
        return False, "Please make sure all the values are numeric"

    # relative abundance
    in_data = in_data / in_data.apply(lambda x: sum(x), axis=0)
    tol = 1e-6
    if all(abs(in_data.sum(axis=0) - 1) > tol):
        return False, "Input must be a vector of absolute or relative abundance"

    in_data = in_data.fillna(0)

    # check order
    if not str(q).isnumeric() and str(q) != "inf":
        return False, "q must be numeric"

    if q == 0 and keep:
        return in_data.apply(lambda x: sum(x > 0), axis=0)

    else:
        if q == 0:
            return in_data.apply(lambda x: sum(x > 0), axis=0)

        elif q == 1:
            p = in_data.copy()
            w = np.log(in_data)
            pw = p * w
            return np.exp(-pw.sum(axis=0))

        elif q == "inf":
            return 1 / in_data.apply(lambda x: max(x), axis=0)

        else:
            p = in_data.copy()
            w = np.power(p, q - 1)
            return np.power((p * w).sum(axis=0), (1 / (1 - q)))


def shannon_entropy(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return
    df = df / df.sum(axis=0)
    shannon_df = (-1) * df * np.log(df)
    return shannon_df.sum(axis=0)


def ginni_simpson(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return
    df = df / df.sum(axis=0)
    ginni_df = df ** 2
    return 1 - ginni_df.sum(axis=0)


def _simpson_helper(x):
    x = x[x > 0]
    N = sum(x)
    cum_sum = sum(n * (n - 1) for n in x)
    return 1 - (cum_sum / (N * (N - 1)))


def simpson(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return

    return df.apply(lambda x: _simpson_helper(x), axis=0)


def inverse_simpson(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return
    df = df / df.sum(axis=0)
    inverse_df = df ** 2
    return 1 / inverse_df.sum(axis=0)


def observed_richness(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return
    df = df / df.sum(axis=0)
    return df.apply(lambda x: sum(x > 0), axis=0)


def chao1(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return

    f1 = df.apply(lambda x: sum(x == 1), axis=0)
    f2 = df.apply(lambda x: sum(x == 2), axis=0)
    s = df.apply(lambda x: sum(x > 0), axis=0)
    for i, k in f1.iteritems():
        if f1[i] != 0 and f2[i] != 0:
            s[i] = s[i] + (f1[i] * (f1[i] - 1)) / (2 * (f2[i] + 1))

    return s


def first_order_jackknife(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return

    f1 = df.apply(lambda x: sum(x == 1), axis=0)
    s = df.apply(lambda x: sum(x > 0), axis=0)
    return s + f1


def second_order_jackknife(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return

    f1 = df.apply(lambda x: sum(x == 1), axis=0)
    f2 = df.apply(lambda x: sum(x == 2), axis=0)
    s = df.apply(lambda x: sum(x > 0), axis=0)
    return s + 2 * f1 - f2


def abundance_coverage_estimator(df):
    flag, df = reformat_file(df)
    if not flag:
        print(df)
        return

    s_rare = df.apply(lambda x: sum(x <= 10), axis=0)
    s_abund = df.apply(lambda x: sum(x > 10), axis=0)
    f1 = df.apply(lambda x: sum(x == 1), axis=0)
    n_rare = sum(df.apply(lambda x: sum(x == i), axis=0) for i in range(1, 11))
    c_ace = 1 - (f1 / n_rare)
    gamma_ace = sum(i * (i - 1) * df.apply(lambda x: sum(x == i)) for i in range(1, 11)) * s_rare / (
            c_ace * n_rare * (n_rare - 1)) - 1

    s_ace = s_abund + (s_rare / c_ace)
    return "Coming soon...."


def hill_numbers(df, q, keep=False):
    return qd(df, q, keep)


def berger_parker(df, keep=False):
    return 1 / qd(df, "inf", keep)


def renyi_entropy(df, q, keep=False):
    return np.log(qd(df, q, keep))


def tail(df):
    def helper(x):
        p = x[x > 0]
        p = p.sort_values(ascending=False)
        p = p[1:]
        return np.sqrt(sum((i ** 2) * p[i] for i in range(1, len(p))))

    df = df / df.apply(lambda x: sum(x), axis=0)
    return df.apply(lambda x: helper(x), axis=0)


def evenness_factor(df, q, keep=False):
    return qd(df, q, keep) / qd(df, 0, keep)


def inverse_evenness_factor(df, q, keep=False):
    return qd(df, 0, keep) / qd(df, q, keep)


def relative_evenness(df, q, keep=False):
    return np.log(qd(df, q, keep)) / np.log(qd(df, 0, keep))


def inverse_relative_evenness(df, q, keep=False):
    return 1 - np.log(qd(df, q, keep)) / np.log(qd(df, 0, keep))


def Pielou(df, keep=False):
    return np.log(qd(df, 1, keep)) / np.log(qd(df, 0, keep))
