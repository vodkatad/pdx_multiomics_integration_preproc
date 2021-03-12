import numpy as np
import pandas as pd
import numba as nb
from numba.typed import List
from itertools import combinations


@nb.jit(nopython=True, parallel=True)
def pcc(data1, data2):
    M = data1.size

    sum1 = 0.
    sum2 = 0.
    for i in nb.prange(M):
        sum1 += data1[i]
        sum2 += data2[i]
    mean1 = sum1 / M
    mean2 = sum2 / M

    var_sum1 = 0.
    var_sum2 = 0.
    cross_sum = 0.
    for i in nb.prange(M):
        var_sum1 += (data1[i] - mean1) ** 2
        var_sum2 += (data2[i] - mean2) ** 2
        cross_sum += (data1[i] * data2[i])

    std1 = (var_sum1 / M) ** .5
    std2 = (var_sum2 / M) ** .5
    cross_mean = cross_sum / M
    return (cross_mean - mean1 * mean2) / (std1 * std2)


@ nb.jit
# Iterate through the correlation matrix and compare correlations
def filter_feature_coeff(corr_in, features,
                         threshold, priority_features):
    drop_cols = np.array([], dtype="object")
    log = np.array([], dtype="object")

    for i, j in combinations(range(len(features)), 2):
        val = abs(pcc(corr_in[:, i], corr_in[:, j]))
        # If correlation exceeds the threshold
        if val >= threshold:
            f1 = features[i]
            f2 = features[j]
            # if both features in priority set
            if f1 in priority_features and f2 in priority_features:
                f_todrop = f1  # drop f1
            else:
                # drop the one not in priority set or
                # the first one if both are not in priority set
                f_todrop = np.array([f for f in [f1, f2]
                                     if f not in priority_features])[0]
            np.append(drop_cols, f_todrop)
            out = np.array([f1, f2, val,
                            f1 in priority_features,
                            f2 in priority_features,
                            f_todrop])
            np.append(log, out)
    return drop_cols, log


def remove_collinear_features_numba(x, threshold,
                                    priority_features=[],
                                    logfile=None):
    # Calculate the correlation matrix
    corr_in = x.fillna(0).astype(float).values
    drop_cols, log = filter_feature_coeff(
        corr_in, x.columns, threshold, priority_features)
    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    if logfile is not None:
        log_cols = ["f1", "f2", "corr",
                    "f1_priority", "f2_priority", "dropped"]
        pd.DataFrame(log, columns=log_cols).to_csv(logfile, sep="\t")
    return x


def remove_collinear_features(x, threshold,
                              priority_features=[],
                              logfile=None):
    '''
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
    '''
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    log = []
    log_cols = ["f1", "f2", "corr", "f1_priority", "f2_priority", "dropped"]
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j: (j+1), (i+1): (i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)[0][0]
            # If correlation exceeds the threshold
            if val >= threshold:
                f1 = col.values[0]
                f2 = row.values[0]
                # if both features in priority set
                if f1 in priority_features and f2 in priority_features:
                    drop_cols.append(f1)
                    log.append([f1, f2, val, True, True, f1])
                else:
                    f_todrop = [f for f in [f1, f2]
                                if f not in priority_features][0]
                    drop_cols.append(f1)
                    log.append([f1, f2, val,
                                f1 in priority_features,
                                f2 in priority_features,
                                f_todrop])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    if logfile is not None:
        pd.DataFrame(log, columns=log_cols).to_csv(logfile, sep="\t")
    return x
