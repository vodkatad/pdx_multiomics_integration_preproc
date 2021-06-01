#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import label_binarize

# model presistence
import pickle

# feature selection
from mlxtend.feature_selection import ColumnSelector
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold, chi2

# model tuning
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold

# model validation
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report

# models
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

target_col = snakemake.params.target_col

f = snakemake.input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='Cetuximab')))]

f = snakemake.input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='Cetuximab')))]
Expr.columns = [c + "_expr" for c in Expr.columns]

f = snakemake.input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='Cetuximab')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]

f = snakemake.input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='Cetuximab')))]
Mut.columns = [c + "_mut" for c in Mut.columns]

f = snakemake.input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target
Y_class_dict = {'PD': 0, 'SD': 1, 'OR': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

# outer merge 'omics on PD model ID
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
feature_col = all_df.columns.tolist()
all_df = all_df.select_dtypes([np.number])
# merge with target variable (drug response) on PDx ID
all_df = pd.merge(all_df, Y[target_col],
                  right_index=True, left_index=True, how="right")

# fillna in features with median imputation
all_df[feature_col] = all_df[feature_col].astype(
    float).apply(lambda col: col.fillna(col.median()))
# drop duplicated instances (on PDx id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]

# scale features [0-1]
all_df = pd.DataFrame(MinMaxScaler().fit_transform(all_df.values),
                      columns=all_df.columns,
                      index=all_df.index)

# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col].values
y_train = all_df.loc[train_models, target_col].values
X_test = all_df.loc[test_models, feature_col].values
y_test = all_df.loc[test_models, target_col].values

# get col indeces for feature subsets, one per OMIC
Meth_indeces = list(range(0, Meth.shape[1]))
pos = len(Meth_indeces)
Expr_indeces = list(range(Meth_indeces[-1]+1, pos + Expr.shape[1]))
pos += len(Expr_indeces)
Mut_indeces = list(range(Expr_indeces[-1]+1, pos + Mut.shape[1]))
pos += len(Mut_indeces)
CNV_indeces = list(range(Mut_indeces[-1]+1, pos + Mut.shape[1]))

# pipeline to train a classifier on methylation data alone
pipe_steps = [
    # select only meth features
    ("ColumnSelector", ColumnSelector(cols=Meth_indeces)),
    # drop features with 0 variance
    ("VarianceFilter", VarianceThreshold(threshold=0)),
    # univariate filter on chi2 stat
    ("chi2filterFscore", SelectKBest(chi2)),
    # level 1 classifier
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeMeth = Pipeline(pipe_steps)

# pipeline to train a classifier on expression data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Expr_indeces)),
    ("VarianceFilter", VarianceThreshold(threshold=0)),
    # feature selection by importance in Extra Tree classifier
    ("ExtraTreesSelector", SelectFromModel(
        ExtraTreesClassifier(criterion='gini', class_weight='balanced'))),
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced'))
]
pipeExpr = Pipeline(pipe_steps)

# pipeline to train a classifier on mutation data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Mut_indeces)),
    ("VarianceFilter", VarianceThreshold(threshold=0)),
    ("chi2filterFscore", SelectKBest(chi2)),
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeMut = Pipeline(pipe_steps)

# pipeline to train a classifier on CNV data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=CNV_indeces)),
    ("VarianceFilter", VarianceThreshold(threshold=0)),
    ("chi2filterFscore", SelectKBest(chi2)),
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeCNV = Pipeline(pipe_steps)

# build the meta classifier
sclf = StackingCVClassifier(classifiers=[pipeMeth,
                                         pipeExpr,
                                         pipeMut,
                                         pipeCNV],
                            cv=3,
                            random_state=13,
                            verbose=1,
                            # use_probas=True, #average_probas=False,
                            meta_classifier=LogisticRegression())

hyperparameter_grid = {
    # Meth params
    'pipeline-1__chi2filterFscore__k': [80],
    'pipeline-1__RFClassifier__n_estimators': [15],
    'pipeline-1__RFClassifier__max_depth': [4],
    'pipeline-1__RFClassifier__min_samples_leaf': [15],
    'pipeline-1__RFClassifier__max_features': [.4, ],
    'pipeline-1__RFClassifier__min_samples_split': [.4, ],
    # Expr params
    'pipeline-2__ExtraTreesSelector__estimator__max_features': [.3, ],
    'pipeline-2__ExtraTreesSelector__estimator__min_samples_split': [5, ],
    'pipeline-2__ExtraTreesSelector__estimator__min_samples_leaf': [15],
    'pipeline-2__RFClassifier__min_samples_leaf': [10, 12],
    'pipeline-2__RFClassifier__n_estimators': [80],
    'pipeline-2__RFClassifier__max_depth': [16],
    # Mut params
    'pipeline-3__chi2filterFscore__k': [55, 'all'],
    'pipeline-3__RFClassifier__max_features': [.2],
    'pipeline-3__RFClassifier__min_samples_split': [.4],
    'pipeline-3__RFClassifier__n_estimators': [15],
    'pipeline-3__RFClassifier__max_depth': [11],
    # CNV params
    'pipeline-4__chi2filterFscore__k': [165, 'all'],
    'pipeline-4__RFClassifier__max_features': [.4],
    'pipeline-4__RFClassifier__min_samples_split': [0.095],
    'pipeline-4__RFClassifier__max_depth': [4],
    'pipeline-4__RFClassifier__n_estimators': [25],
    # 'pipeline-4__' : [],
    'meta_classifier__C':  np.linspace(.01, .95, 10, endpoint=True),
}

# Set up the random search with 4-fold stratified cross validation
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
grid = GridSearchCV(estimator=sclf,
                    param_grid=hyperparameter_grid,
                    n_jobs=-1,
                    cv=skf,
                    refit=True,
                    verbose=2)
grid.fit(X_train, y_train)

# run grid search and print params, score to log
cv_keys = ('mean_test_score', 'std_test_score', 'params')
arr = []
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    arr.append([grid.cv_results_[cv_keys[0]][r],
                grid.cv_results_[cv_keys[1]][r] / 2.0,
                grid.cv_results_[cv_keys[2]][r]])
grid_df = pd.DataFrame(arr, columns=cv_keys)
grid_df.to_csv(snakemake.log.grid, sep="\t")

# pickle best model from gridCV
model_filename = snakemake.output.model
with open(model_filename, 'wb') as f:
    pickle.dump(grid.best_estimator_, f)

# print best model params to log
with open(snakemake.log.bestClassifier, "w") as f:
    f.write('Best parameters: %s' % grid.best_params_)
    f.write('Accuracy: %.2f' % grid.best_score_)
    classifier = grid.best_estimator_
    # assess best classifier performance on test set
    grid_test_score = classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    f.write(f'Accuracy on test set: {grid_test_score:.3f}')
    # print classification report on test set
    f.write(classification_report(y_test, y_pred,
                                  target_names=['PD', 'SD-OR']))
