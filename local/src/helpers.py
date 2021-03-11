import pandas as pd


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
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
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
