#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Train, optimise stacked predictor of Cetuximab sensitivity
import pandas as pd
import numpy as np
import warnings
# optuna + visulization
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import plotly.graph_objects as go
import ipywidgets as ipyw

from catboost import CatBoostClassifier
import pickle
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_curve,  auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold, chi2, SelectFromModel
from sklearn import model_selection
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import no_type_check_decorator

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 600
pd.options.display.max_rows = 30


# # Analysis/Modeling
# Do work here

input = snakemake.input
output = snakemake.output
target_col = snakemake.params.target_col


# Analysis/Modeling
# load all 'omics preprocessed datasets
# K5 clusters encoded meth probes
f = input.meth
Meth = pd.read_csv(f, sep="\t", header=0, index_col=0)
Meth = Meth[Meth.columns.drop(list(Meth.filter(regex='Cetuximab')))]
# encoded expr data w/t progeny pathway scores + msdb hallmarks ssGSEA scores
# processed through a colinearity + chi2 filter (drop the worst of each colinear pair of features)
f = input.expr
Expr = pd.read_csv(f, sep="\t", header=0, index_col=0)
Expr = Expr[Expr.columns.drop(list(Expr.filter(regex='Cetuximab')))]
Expr.columns = [c + "_expr" for c in Expr.columns]
# feature agglomeration CNV, input includes highGain features (> than 1 copy gained)
f = input.cnv
CNV = pd.read_csv(f, sep="\t", header=0, index_col=0)
CNV = CNV[CNV.columns.drop(list(CNV.filter(regex='Cetuximab')))]
CNV.columns = [c + "_cnv" for c in CNV.columns]
# custom mut feature cross w/t top 20 features by chi2
f = input.mut
Mut = pd.read_csv(f, sep="\t", header=0, index_col=0)
Mut = Mut[Mut.columns.drop(list(Mut.filter(regex='Cetuximab')))]
Mut.columns = [c + "_mut" for c in Mut.columns]
# add clinical data (custom encoding, filtering)
f = input.clin
Clin = pd.read_csv(f, sep="\t", header=0, index_col=0)
Clin = Clin[Clin.columns.drop(list(Clin.filter(regex='Cetuximab')))]
Clin.columns = [c + "_clin" for c in Clin.columns]
# load drug response data
f = input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target var (binary responder/non-responder)
Y_class_dict = {'PD': 0, 'OR+SD': 1}
Y[target_col] = Y[target_col].replace(Y_class_dict)

# merge all feature blocks + response together
df1 = pd.merge(Mut, CNV, right_index=True, left_index=True, how="outer")
df2 = pd.merge(Meth, Expr, right_index=True, left_index=True, how="outer")
all_df = pd.merge(df2, df1, right_index=True, left_index=True, how="outer")
all_df = pd.merge(all_df, Clin, right_index=True, left_index=True, how="outer")

feature_col = all_df.columns.tolist()
nonbin_features = [c for c in all_df if all_df[c].nunique() > 2]
all_df = pd.merge(all_df, Y[target_col],
                  right_index=True, left_index=True, how="right")
# drop duplicated instances (ircc_id) from index
all_df = all_df[~all_df.index.duplicated(keep='first')]
# fill sparse features with median imputation
all_df[feature_col] = all_df.fillna(0)
# force to numeric
#all_df = all_df.select_dtypes([np.number])
# train-test split
train_models = Y[Y.is_test == False].index.unique()
test_models = Y[Y.is_test == True].index.unique()
X_train = all_df.loc[train_models, feature_col]
y_train = all_df.loc[train_models, target_col]
X_test = all_df.loc[test_models, feature_col]
y_test = all_df.loc[test_models, target_col]
# scale features separately
scaler = MinMaxScaler().fit(X_train)
X_train[nonbin_features] = scaler.transform(X_train[nonbin_features].values)
scaler = MinMaxScaler().fit(X_test)
X_test[nonbin_features] = scaler.transform(X_test[nonbin_features].values)
# log train, test shape, dataset balance
logfile = 'stacked_input.log'
with open(logfile, "w") as log:
    log.write(
        f"There are {X_train.shape[0]} instances in the trainig set." + '\n')
    log.write(f"There are {X_test.shape[0]} instances in the test set." + '\n')
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    log.write(
        f"There are {train_counts.loc[0]} 'PD' instances and {train_counts.loc[1]} 'SD+OR' instances in the training set." + '\n')
    log.write(
        f"There are {test_counts.loc[0]} 'PD' instances and {test_counts.loc[1]} 'SD+OR' instances in the test set." + '\n')
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

# log n of features for each block
logfile = snakemake.log[0]
with open(logfile, "a") as log:
    log.write(f"There are {X_train.shape[1]} total features." + '\n')
    log.write(f"There are {Meth.shape[1]} methylation features." + '\n')
    log.write(
        f"There are {Expr.shape[1]} Hallmarks, PROGENy expression features." + '\n')
    log.write(f"There are {Mut.shape[1]} mutation features." + '\n')
    log.write(f"There are {CNV.shape[1]} copy number features." + '\n')
    log.write(f"There are {Clin.shape[1]} clinical features." + '\n')

# save train, test to file
X_test.to_csv(output.X_test, sep='\t')
X_train.to_csv(output.X_train, sep='\t')
y_test.to_csv(output.Y_test, sep='\t')
y_train.to_csv(output.Y_train, sep='\t')


def calc_variance(X, Y):
    return pd.DataFrame(X).var()


fitted_models = []
# objective function for optuna


def objective(trial):
    # parameters to optimize
    Meth_KNNlassifier__n_neighbors = trial.suggest_int(
        "Meth_KNNlassifier__n_neighbors", 5, 25)
    Expr__chi2filterFscore__k = trial.suggest_int(
        "Expr__chi2filterFscore__k", 14, 24)
    Mut__chi2filterFscore__k = trial.suggest_int(
        "Mut__chi2filterFscore__k", 2, 10)
    # trial.suggest_int("CNV__WardAgg__n_clusters", 55, 85, step=5)
    CNV__WardAgg__n_clusters = 75
    CNV__chi2filterFscore__k = trial.suggest_int(
        "CNV__chi2filterFscore__k", 20, 45, )
    Clin__chi2filterFscore__k = trial.suggest_int(
        "Clin__chi2filterFscore__k", 4, 12)
    meta__l1R = trial.suggest_loguniform("meta__C", 0.01, 1)
    #meta__penalty = trial.suggest_categorical("meta__penalty", ['l1', 'l2'])
    #trial.suggest_categorical("meta__use_probas", [True, False])
    #meta__secondary = trial.suggest_categorical("meta__use_secondary", [True, False])
    meta__use_probas = True
    # Mut_rf_n_estimators = trial.suggest_int(
    #    "Mut_rf_n_estimators", 10, 500, log=True)
    #Mut_rf_max_depth = trial.suggest_int("Mut_rf_max_depth", 1, 32, log=True)
    # Expr_rf_n_estimators = trial.suggest_int(
    #    "Expr_rf_n_estimators", 10, 500, log=True)
    Expr_rf_max_depth = trial.suggest_int("Expr_rf_max_depth", 1, 32, log=True)
    # Clin_rf_n_estimators = trial.suggest_int(
    #    "Clin_rf_n_estimators", 10, 500, log=True)
    #Clin_rf_max_depth = trial.suggest_int("Clin_rf_max_depth", 1, 32, log=True)
    CNV_rf_n_estimators = trial.suggest_int(
        "CNV_rf_n_estimators", 10, 500, log=True)
    CNV_rf_max_depth = trial.suggest_int("CNV_rf_max_depth", 1, 32, log=True)
    #Meta__n_estimators = trial.suggest_categorical("Meta__n_estimators", [100, 1000, 10000])
    #Meta__learning_rate=trial.suggest_float("Meta__learning_rate", 0.01, 0.3)
    #Meta__max_depth=trial.suggest_int("Meta__max_depth", 3, 12)

    # build stacked model pipeline
    # pipeline to train a classifier on meth data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Meth_indeces)),
        ('KNNlassifier', KNeighborsClassifier(
            n_neighbors=Meth_KNNlassifier__n_neighbors)),
    ]
    pipeMeth = Pipeline(pipe_steps)

    # pipeline to train a classifier on Hallmarks, PROGENy scores (expression)
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Expr_indeces)),
        ("chi2filterFscore", SelectKBest(chi2, k=Expr__chi2filterFscore__k)),
        ('RFClassifier', RandomForestClassifier(max_depth=Expr_rf_max_depth))]
    pipeExpr = Pipeline(pipe_steps)

    # pipeline to train a classifier on mutation data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Mut_indeces)),
        # univariate filter on chi2 stat
        ("chi2filterFscore", SelectKBest(chi2, k=Mut__chi2filterFscore__k)),
        ('RFClassifier', CatBoostClassifier())  # RandomForestClassifier(
        # max_depth=Mut_rf_max_depth, n_estimators=Mut_rf_n_estimators)),
    ]
    pipeMut = Pipeline(pipe_steps)

    # pipeline to train a classifier on CNV data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=CNV_indeces)),
        # remove samples which have the same val in 5% or more samples
        ("VarianceFilter", VarianceThreshold(threshold=(.75 * (1 - .75)))),
        # Ward feature agglomeration by mean
        ("WardAgg", FeatureAgglomeration(n_clusters=CNV__WardAgg__n_clusters)),
        ("chi2filterFscore", SelectKBest(chi2, CNV__chi2filterFscore__k)),
        ('RFClassifier', RandomForestClassifier(
            max_depth=CNV_rf_max_depth, n_estimators=CNV_rf_n_estimators)),
    ]
    pipeCNV = Pipeline(pipe_steps)

    # pipeline to train a classifier on clinical/patient data alone
    pipe_steps = [
        ("ColumnSelector", ColumnSelector(cols=Clin_indeces)),
        ("chi2filterFscore", SelectKBest(chi2, k=Clin__chi2filterFscore__k)),
        ('RFClassifier', RandomForestClassifier()),
    ]
    pipeClin = Pipeline(pipe_steps)

    # build the meta classifier
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=13)
    sclf = StackingCVClassifier(classifiers=[
        pipeMeth,
        pipeExpr,
        pipeMut,
        pipeCNV,
        pipeClin
    ],
        cv=skf,
        n_jobs=snakemake.threads,
        shuffle=True,
        random_state=13,
        verbose=0,
        use_probas=True,
        # average_probas=False,
        # use_features_in_secondary=meta__secondary,
        meta_classifier=LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=meta__l1R))

    # fit on train, test return ROC AUC
    sclf = sclf.fit(X_train, y_train)
    #train_auc = roc_auc_score(y_train, sclf.predict(X_train))
    train_auc = cross_val_score(
        sclf, X_train, y_train, scoring='roc_auc',
        n_jobs=snakemake.threads, cv=2).mean()
    test_auc = roc_auc_score(y_test, sclf.predict(X_test))
    fitted_models.append([test_auc, train_auc, sclf])
    return train_auc


CS_study = optuna.create_study(
    study_name='Cetuximab_sensitivity_prediction',
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(max_resource="auto")
)

CS_study.optimize(objective, n_trials=snakemake.threads,
                  n_jobs=snakemake.threads)


# pickle best model from trials
best_model_df = pd.DataFrame(fitted_models,
                             columns=['test_auc', 'train_auc', 'fitted_model_obj']).\
    sort_values('train_auc', ascending=False)
best_model = best_model_df.iloc[0].fitted_model_obj
model_filename = snakemake.output.best_model
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)
