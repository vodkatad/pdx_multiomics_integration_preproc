#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Train, optimise stacked predictor of Cetuximab sensitivity

# ### Imports
# Import libraries and write settings here.
# Data manipulation
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve, multilabel_confusion_matrix, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, chi2
from sklearn import model_selection
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import no_type_check_decorator
import pandas as pd
import numpy as np
import warnings
# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30
# scalers
# models
# processing
# feature selection
# feature agglomeration
# benchmark
# Hyperparameter tuning
# model persistence


# Analysis/Modeling
# load all 'omics preprocessed datasets
# K5 clusters encoded meth probes
f = snakemake.input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='Cetuximab')))]
Meth.columns = ["cluster_"+c+"_meth" for c in Meth.columns]
# encoded expr data w/t progeny pathway scores + msdb hallmarks ssGSEA scores
# processed through a colinearity + chi2 filter (drop the worst of each colinear pair of features)
f = snakemake.input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='Cetuximab')))]
Expr.columns = [c + "_expr" for c in Expr.columns]

# raw methylation probes data
f = snakemake.input.raw_meth
rawMeth = pd.read_csv(f, sep="\t", header=0, index_col=0)
rawMeth = rawMeth[rawMeth.columns.drop(
    list(rawMeth.filter(regex='Cetuximab')))]
rawMeth.columns = [c+"_rawMeth" for c in rawMeth.columns]
# raw expression data (variance-stabilised RNAseq)
f = snakemake.input.raw_expr
rawExpr = pd.read_csv(f, sep="\t", header=0, index_col=0)
rawExpr = rawExpr[rawExpr.columns.drop(
    list(rawExpr.filter(regex='Cetuximab')))]
rawExpr.columns = [c + "_rawExpr" for c in rawExpr.columns]

# feature agglomeration CNV, input includes highGain features (> than 1 copy gained)
f = snakemake.input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='Cetuximab')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]
# custom mut feature cross w/t top 20 features by chi2
f = snakemake.input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='Cetuximab')))]
Mut.columns = [c + "_mut" for c in Mut.columns]
# add clinical data (custom encoding, filtering)
f = snakemake.input.clin
Clin = pd.read_csv(f, sep="\t", header=0, index_col=0)
Clin = Clin[Clin.columns.drop(list(Clin.filter(regex='Cetuximab')))]
Clin.columns = [c + "_clin" for c in Clin.columns]
# load drug response data
f = snakemake.input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target var (binary responder/non-responder)
target_col = snakemake.params.target_col
Y_class_dict = {'PD': 0, 'OR+SD': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

# merge all feature blocks + response together
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Clin, right_index=True,
                  left_index=True, how="outer")
all_df = pd.merge(all_df, rawExpr, right_index=True,
                  left_index=True, how="outer")
all_df = pd.merge(all_df, rawMeth, right_index=True,
                  left_index=True, how="outer")
all_df = all_df[all_df.columns.drop(
    list(all_df.filter(regex='ircc_id')))]
feature_col = all_df.columns.tolist()
all_df = all_df.select_dtypes([np.number])
all_df = pd.merge(all_df, Y[target_col],
                  right_index=True, left_index=True, how="right")
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]
# fill sparse features with median imputation
all_df[feature_col] = all_df[feature_col].\
    astype(float).apply(lambda col: col.fillna(col.median()))
# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col]
y_train = all_df.loc[train_models, target_col]
X_test = all_df.loc[test_models, feature_col]
y_test = all_df.loc[test_models, target_col]
# scale features separately
scaler = MinMaxScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train.values),
                       columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test.values),
                      columns=X_test.columns, index=X_test.index)
# log train, test shape, dataset balance
logfile = snakemake.log[0]
with open(logfile, "w") as log:
    log.write(
        f"There are {X_train.shape[0]} instances in the trainig set." + '\n')
    log.write(f"There are {X_test.shape[0]} instances in the test set." + '\n')
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    log.write(f"There are {train_counts.loc[0]} 'PD' instances and\
         {train_counts.loc[1]} 'SD+OR' instances in the training set." + '\n')
    log.write(f"There are {test_counts.loc[0]} 'PD' instances and\
         {test_counts.loc[1]} 'SD+OR' instances in the test set." + '\n')
# get indeces for feature subsets, one per OMIC
Meth_indeces = list(range(0, Meth.shape[1]))
pos = len(Meth_indeces)
Expr_indeces = list(range(Meth_indeces[-1]+1, pos + Expr.shape[1]))
pos += len(Expr_indeces)
Mut_indeces = list(range(Expr_indeces[-1]+1, pos + Mut.shape[1]))
pos += len(Mut_indeces)
CNV_indeces = list(range(Mut_indeces[-1]+1, pos + CNV.shape[1]))
pos += len(CNV_indeces)
Clin_indeces = list(range(CNV_indeces[-1]+1, pos + Clin.shape[1]))
pos += len(Clin_indeces)
rawExpr_indeces = list(range(Clin_indeces[-1]+1, pos + rawExpr.shape[1]))
pos += len(rawExpr_indeces)
rawMeth_indeces = list(range(rawExpr_indeces[-1]+1, pos + rawMeth.shape[1]))

# log n of features for each block
with open(logfile, "a") as log:
    log.write(f"There are {X_train.shape[1]} total features." + '\n')
    log.write(f"There are {Meth.shape[1]} methylation features." + '\n')
    log.write(f"There are {Expr.shape[1]} expression features." + '\n')
    log.write(f"There are {Mut.shape[1]} mutation features." + '\n')
    log.write(f"There are {CNV.shape[1]} copy number features." + '\n')
    log.write(f"There are {Clin.shape[1]} clinical features." + '\n')
    log.write(f"There are {rawExpr.shape[1]} raw expression features." + '\n')
    log.write(f"There are {rawMeth.shape[1]} raw methylation features." + '\n')
# write train,test sets to file
X_train.to_csv(snakemake.output.X_train, sep='\t')
X_test.to_csv(snakemake.output.X_test, sep='\t')

# build stacked model pipeline

# pipeline to train a classifier on clustered meth data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Meth_indeces)),
    # drop features with 0 variance
    ("VarianceFilter", VarianceThreshold(threshold=0)),
    ('KNNlassifier', KNeighborsClassifier()),
]
pipeMeth = Pipeline(pipe_steps)

# pipeline to train a classifier on encoded expression data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Expr_indeces)),
    ("chi2filterFscore", SelectKBest(chi2)),
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeExpr = Pipeline(pipe_steps)

# pipeline to train a classifier on raw expression data alone
# L1-penalised (Lasso) multivariate fature selection
L1LR = LogisticRegression(
    penalty='l1',
    solver='saga')
# single-omic elastic net classifier
elasticNetClassifier = LogisticRegression(penalty='elasticnet',
                                          random_state=13,
                                          solver='saga')
L1Selector = SelectFromModel(estimator=L1LR)
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=rawExpr_indeces)),
    ("L1Selector", L1Selector),
    ('elasticNetClassifier', elasticNetClassifier),
]
pipeRawExpr = Pipeline(pipe_steps)

# pipeline to train a classifier on raw methylation data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=rawMeth_indeces)),
    ("L1Selector", L1Selector),
    ('elasticNetClassifier', elasticNetClassifier),
]
pipeRawMeth = Pipeline(pipe_steps)

# pipeline to train a classifier on mutation data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Mut_indeces)),
    ("chi2filterFscore", SelectKBest(chi2)),  # univariate filter on chi2 stat
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeMut = Pipeline(pipe_steps)

# pipeline to train a classifier on CNV data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=CNV_indeces)),
    # remove samples which have the same val in 85% or more samples
    ("VarianceFilter", VarianceThreshold(threshold=(.75 * (1 - .75)))),
    ("WardAgg", FeatureAgglomeration()),  # Ward feature agglomeration by mean
    ("chi2filterFscore", SelectKBest(chi2)),
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeCNV = Pipeline(pipe_steps)

# pipeline to train a classifier on clinical/patient data alone
pipe_steps = [
    ("ColumnSelector", ColumnSelector(cols=Clin_indeces)),
    ("chi2filterFscore", SelectKBest(chi2)),
    ('RFClassifier', RandomForestClassifier(
        criterion='gini', class_weight='balanced')),
]
pipeClin = Pipeline(pipe_steps)

# build the meta classifier
LRL2_classifier = LogisticRegression(penalty='l2',
                                     class_weight='balanced')
sclf = StackingCVClassifier(classifiers=[pipeMeth, pipeExpr,
                                         pipeRawExpr, pipeRawMeth, pipeMut,
                                         pipeCNV, pipeClin],
                            cv=3, random_state=13, verbose=1,
                            # use_probas=True, #average_probas=False,
                            # use_features_in_secondary=True,
                            meta_classifier=elasticNetClassifier)

hyperparameter_grid = {
    # Meth params
    'pipeline-1__KNNlassifier__n_neighbors': [13],
    'pipeline-1__KNNlassifier__p': [1],
    # 'pipeline-1__': [],
    # Expr params
    'pipeline-2__chi2filterFscore__k': [25, 'all'],
    'pipeline-2__RFClassifier__max_features': [.2],
    'pipeline-2__RFClassifier__min_samples_split': [.4],
    'pipeline-2__RFClassifier__n_estimators': [15],
    'pipeline-2__RFClassifier__max_depth': [11],
    # rawExpr pipeline
    'pipeline-3__L1Selector__estimator__C': np.linspace(5, 70, 3, endpoint=True),
    'pipeline-3__elasticNetClassifier__l1_ratio': np.linspace(.01, .9, 3, endpoint=True),
    'pipeline-4__L1Selector__estimator__C': np.linspace(5, 70, 3, endpoint=True),
    'pipeline-4__elasticNetClassifier__l1_ratio': np.linspace(.01, .9, 3, endpoint=True),
    # Mut params
    'pipeline-5__chi2filterFscore__k': [50, 'all'],
    'pipeline-5__RFClassifier__max_features': [.2],
    'pipeline-5__RFClassifier__min_samples_split': [.01],
    'pipeline-5__RFClassifier__n_estimators': [15],
    'pipeline-5__RFClassifier__max_depth': [11],
    # 'pipeline-3__' : [],
    # CNV params
    'pipeline-6__WardAgg__n_clusters': [40],
    'pipeline-6__chi2filterFscore__k': [12, 'all'],
    'pipeline-6__RFClassifier__max_features': [.4],
    'pipeline-6__RFClassifier__min_samples_split': [.4],
    'pipeline-6__RFClassifier__n_estimators': [10],
    'pipeline-6__RFClassifier__max_depth': [3],
    # clin params
    'pipeline-7__chi2filterFscore__k': [10],
    'pipeline-7__chi2filterFscore__k': [25, 'all'],
    'pipeline-7__RFClassifier__max_depth': [4],
    # meta classifier params
    'meta_classifier__l1_ratio':  np.linspace(.01, .9, 5, endpoint=True)
}

# Set up the random search with 4-fold stratified cross validation
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
grid = GridSearchCV(estimator=sclf,
                    param_grid=hyperparameter_grid,
                    n_jobs=-1,
                    cv=skf,
                    refit=True,
                    verbose=2)
# train the stacked model
grid.fit(X_train, y_train)

# log model perfoormance across training CV iterations
cv_keys = ('mean_test_score',
           'std_test_score',
           'params')
logfile = snakemake.log[1]
with open(logfile, "w") as log:
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        log.write("%0.3f +/- %0.2f %r"
                  % (grid.cv_results_[cv_keys[0]][r],
                     grid.cv_results_[cv_keys[1]][r] / 2.0,
                     grid.cv_results_[cv_keys[2]][r]) + '\n')

# log summary of best model params, performance on traing set
logfile = snakemake.log[2]
with open(logfile, "w") as log:
    log.write('Best parameters: %s' % grid.best_params_ + '\n')
    log.write('Accuracy: %.2f' % grid.best_score_ + '\n')

# pickle best model from gridCV
model_filename = snakemake.output.best_model
with open(model_filename, 'wb') as f:
    pickle.dump(grid.best_estimator_, f)

# load the model from file
#classifier = pickle.load(open(model_filename, 'rb'))
# assess best classifier performance on test set
#grid_test_score = classifier.score(X_test, y_test)
#y_pred = classifier.predict(X_test)
#print(f'Accuracy on test set: {grid_test_score:.3f}')

# print classification report on test set
#print(classification_report(y_test, y_pred, target_names=['PD', 'SD-OR']))

##confusion_matrix = confusion_matrix(y_test, y_pred)
# plot_confusion_matrix(classifier, X_test, y_test,
    #display_labels=['PD', 'SD-OR'],
    # cmap=plt.cm.Blues)

# return the marginal probability that the given sample has the label in question
#y_test_predict_proba = classifier.predict_proba(X_test)
#roc_auc_score(y_test, y_test_predict_proba[:, 1])
