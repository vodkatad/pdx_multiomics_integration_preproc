import numpy as np
import pandas as pd
import numba as nb
from numba.typed import List
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import warnings


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
        # pd.DataFrame(log, columns=log_cols).to_csv(logfile, sep="\t")
    return x


def combine_binary_features(X, threshold, max_combine=5):
    # takes in a matrix of binary features and returns a shrinked version
    # by combining by product those features that exceed a given similarity threshold

    X_combine = X.fillna(0).copy()
    # check that all input cols are binary
    bin_cols = [c for c in X_combine.columns if
                X_combine[c].dropna().value_counts().index.isin([0, 1]).all()]
    non_binary_cols = [c for c in bin_cols if c not in X_combine.columns]
    if len(non_binary_cols) > 0:
        warnings.warn(
            f'input columns {non_binary_cols} are not binary, returning original input')
        return X_combine

    def calc_feature_dist(df):
        # calc pairwise cityblock distance b/w binary features
        feature_dist = pd.DataFrame(squareform(pdist(df.T, metric='cityblock')),
                                    index=df.columns,
                                    columns=df.columns).unstack().reset_index()
        feature_dist.columns = ["f1", "f2", "dist"]
        # drop same-same comparisons
        feature_dist = feature_dist[feature_dist.f1 != feature_dist.f2]
        # calc similarity
        feature_dist["similarity"] = feature_dist["dist"] / df.shape[0]
        feature_dist = feature_dist.sort_values("similarity", ascending=False)
        return feature_dist

    def combine_features(row):
        # check if features have already been combined and dropped
        to_drop = [c for c in [row.f1, row.f2] if c in X_combine.columns]
        if len(to_drop) == 2:
            new_feature = row.f1 + "+" + row.f2
            # replace features w/t their product (binary)
            X_combine[new_feature] = X_combine[row.f1] * \
                X_combine[row.f2]
            X_combine.drop(to_drop, inplace=True, axis=1)

    for i in range(max_combine):
        feature_dist = calc_feature_dist(X_combine)
        feature_dist[feature_dist.similarity > threshold].apply(
            combine_features, axis=1)
    return X_combine


def remove_collinear_features(x, threshold,
                              priority_features=[],
                              logfile=None):
    '''
        Remove or combine colinear features in a dataframe with a correlation coefficient
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
