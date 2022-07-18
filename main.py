import numpy as np
import pandas as pd


# assert the index column
# assert the values
# convert to rel abundance if not

def reformat_file(df):
    if not all(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).values):
        return False, "Please make sure all the values are numeric"

    df = df.replace(0, np.nan)
    return True, df


class AlphaIndex:
    def shannon(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return
        df = df / df.sum(axis=0)
        shannon_df = (-1) * df * np.log(df)
        return shannon_df.sum(axis=0)

    def ginni_simpson(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return
        df = df / df.sum(axis=0)
        ginni_df = df ** 2
        return 1 - ginni_df.sum(axis=0)

    def inverse_simpson(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return
        df = df / df.sum(axis=0)
        inverse_df = df ** 2
        return 1 / inverse_df.sum(axis=0)

    def observed_richness(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return
        df = df / df.sum(axis=0)
        return df.apply(lambda x: sum(x > 0), axis=0)

    def chao1(self, df):
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

    def first_order_jackknife(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return

        f1 = df.apply(lambda x: sum(x == 1), axis=0)
        s = df.apply(lambda x: sum(x > 0), axis=0)
        return s + f1

    def second_order_jackknife(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return

        f1 = df.apply(lambda x: sum(x == 1), axis=0)
        f2 = df.apply(lambda x: sum(x == 2), axis=0)
        s = df.apply(lambda x: sum(x > 0), axis=0)
        return s + 2 * f1 - f2

    def abundance_coverage_estimator(self, df):
        flag, df = reformat_file(df)
        if not flag:
            print(df)
            return

        s_rare = df.apply(lambda x: sum(x <= 10), axis=0)
        s_abund = df.apply(lambda x: sum(x > 10), axis=0)
        f1 = df.apply(lambda x: sum(x == 1), axis=0)
        n_rare = sum(i * df.apply(lambda x: sum(x == i), axis=0) for i in range(1, 11))
        c_ace = 1 - (f1 / n_rare)
        gamma_ace = (s_rare / c_ace) * sum(i * (i - 1) * f1 * ((n_rare - 1) / n_rare) for i in range(1, 11))

        s_ace = s_abund + (s_rare / c_ace) + (f1 / c_ace)
        return s_ace


df_ = pd.read_csv('data/simCounts.csv')
df_ = df_.set_index('Unnamed: 0')
print(AlphaIndex().abundance_coverage_estimator(df_))
