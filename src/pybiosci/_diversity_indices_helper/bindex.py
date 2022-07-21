import numpy as np
import pandas as pd


def _beta_index(df, index='w'):
    sample_names = list(df.columns)
    beta_df = pd.DataFrame(index=sample_names, columns=sample_names)
    beta_indices = ["w", "c", "r", "I", "e", "m",
                    "mn", "-2", "co", "cc", "-3", "-3n", "rs", "sim", "z"]

    for i in range(len(sample_names) - 1):
        for j in range(i + 1, len(sample_names)):
            x_data = df[sample_names[i]]
            y_data = df[sample_names[j]]

            if not all(x_data.astype(str).str.isnumeric()) or not all(y_data.astype(str).str.isnumeric()):
                raise ValueError('All values must be absolute or relative abundance!')

            if index not in beta_indices:
                raise ValueError('Please specify one of the following beta indices\n' + ', '.join(beta_indices))

            beta_df.loc[sample_names[i], sample_names[j]] = betaXY(x_data, y_data, index)

    return beta_df


def betaXY(x, y, index):
    xp = x[x > 0].index
    yp = y[y > 0].index
    a = len(set(xp).intersection(set(yp)))
    b = len(set(xp) - set(yp))
    c = len(set(yp) - set(xp))

    if index == 'w':
        return (b + c) / (2 * a + b + c)

    elif index == 'c':
        return -(b + c) / 2

    elif index == 'r':
        return 2 * b * c / ((a + b + c) ** 2 - 2 * b * c)

    elif index == 'I':
        return np.log(2 * a + b + c) - 2 * a * np.log(2) / (2 * a + b + c) - (
                (a + b) * np.log(a + b) + (a + c) * np.log(a + c)) / (2 * a + b + c)

    elif index == 'e':
        return np.exp(np.log(2 * a + b + c) - 2 * a * np.log(2) / (2 * a + b + c) - (
                (a + b) * np.log(a + b) + (a + c) * np.log(a + c)) / (2 * a + b + c)) - 1

    elif index == 'm':
        return (2 * a + b + c) * (b + c) / (a + b + c)

    elif index == 'mn':
        return (2 * a + b + c) * (b + c) / (a + b + c) ** 2

    elif index == '-2':
        return min(b, c) / (a + max(b, c))

    elif index == 'co':
        return (a * c + a * b + 2 * b * c) / (2 * (a + b) * (a + c))

    elif index == 'cc':
        return (b + c) / (a + b + c)

    elif index == '-3':
        return min(b, c) / (a + b + c)

    elif index == '-3n':
        return 2 * min(b, c) / (a + b + c)

    elif index == 'rs':
        return 2 * (b * c + 1) / ((a + b + c) ** 2 - (a + b + c))

    elif index == 'sim':
        return min(b, c) / (min(b, c) + a)

    elif index == 'z':
        return (np.log(2) - np.log(2 * a + b + c) + np.log(a + b + c)) / np.log(2)
