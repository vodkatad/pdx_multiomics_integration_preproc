#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import pandas as pd
import numpy as np
# viz
import seaborn as sb
import matplotlib.pyplot as plt

# scikit-learn
# scalers
from sklearn.preprocessing import StandardScaler
# processing
from sklearn.pipeline import Pipeline
# models
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC, SVC
# feature selection
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif
# benchmark
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, matthews_corrcoef, accuracy_score
# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

import pickle

from helpers import remove_collinear_features_numba
# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# Visualizations
# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)


logfile = snakemake.log[0]
# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")
# load expression data from RNAseq
# these are variance stabilized (vsd)
rnaseq_matrix = pd.read_csv(snakemake.input.expr,
                            sep="\t", header=0, index_col=0)
rnaseq_matrix = rnaseq_matrix.T.reset_index().rename(
    columns={'index': 'ircc_id'})
rnaseq_matrix["ircc_id_short"] = rnaseq_matrix.ircc_id.apply(lambda x: x[0:7])
rnaseq_matrix = rnaseq_matrix.drop("ircc_id", axis=1)

# merge expression and drug response
target_col = snakemake.params.target_col
merge_matrix = pd.merge(rnaseq_matrix,
                        drug_response_data[[
                            target_col,
                            "ircc_id_short",
                            "ircc_id"]],
                        on="ircc_id_short")

# drop instances w/t missing target value
merge_matrix = merge_matrix[~merge_matrix[target_col].isna()]
merge_matrix = merge_matrix.drop(
    "ircc_id_short", axis=1).set_index("ircc_id").drop_duplicates()

input_matrix = merge_matrix
input_matrix.index = input_matrix.index.values
features_col = np.array([c for c in input_matrix.columns if c != target_col])

TT_df = drug_response_data[drug_response_data.ircc_id.isin(input_matrix.index)][
    ["ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

# train-test split
X_train = input_matrix.loc[train_models, features_col]
y_train = input_matrix.loc[train_models, target_col]
X_test = input_matrix.loc[test_models, features_col]
y_test = input_matrix.loc[test_models, target_col]

# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                       columns=features_col,
                       index=X_train.index)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                      columns=features_col,
                      index=X_test.index)
X_train = X_train.values
Y_train = y_train.values
X_test = X_test.values
Y_test = y_test.values

# Expression and methylation data are too high-dimensional to integrate.
# We need to perform feature selection for these two OMICs. I'm using SelectFromModel to select features based on importance.
# I'm using LinearSVC as a model b/c it seems to have the best performance on unselected expression data (i.e. it's able to find the most important features). Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero. When the goal is to reduce the dimensionality of the data to use with another classifier, they can be used along with SelectFromModel to select the non-zero coefficients. In particular, sparse estimators useful for this purpose are the Lasso for regression, and of LogisticRegression and LinearSVC for classification.
#
# Other options:
#     - tree-based feature selection
#     - sequential feature selection

# train the feature selector inside a pipeline that tries to maximise
# classification accuracy on the training set
N = len(features_col)
Ks = [int(f) for f in [N/1000, N/500, N/200, N/100, N/50]]
pipe_steps = [
    ("ANOVAfilter", SelectKBest(f_classif)),
    ("lSVCselector", SelectFromModel(
        LinearSVC(penalty="l1", dual=False, max_iter=5000))),  # feature selection
    ("pca", PCA()),
    ("SVCclassifier", SVC(max_iter=5000)),
]
hyperparameter_grid = {
    "ANOVAfilter__k": Ks,
    "lSVCselector__estimator__C": [1, 0.1, 0.5, .05],
    "pca__n_components": [2],
    "SVCclassifier__kernel": ["linear"],
    "SVCclassifier__C": [1, 0.1, 0.1, 10],
    "SVCclassifier__gamma": ["auto", 1, 0.1, 0.5, 0.01]
}
pipeline = Pipeline(pipe_steps)

# Set up grid search with 4-fold cross validation
grid_cv = GridSearchCV(estimator=pipeline,
                       param_grid=hyperparameter_grid,
                       cv=4,
                       scoring="accuracy",
                       n_jobs=snakemake.threads,
                       refit=True,
                       return_train_score=True)
grid_cv.fit(X_train, y_train)

# grid search params tuning results
CVresult_df = pd.DataFrame(grid_cv.cv_results_)
CVresult_df.sort_values("rank_test_score")[
    ["rank_test_score", "mean_train_score", "mean_test_score"]].head()
CVresult_df.to_csv(snakemake.log[0], sep="\t")

grid_cv_test_score = grid_cv.score(X_test, y_test)

y_classes = input_matrix[target_col].unique().tolist()
Y_pred = grid_cv.predict(X_test)

# if multiclass predictor
if len(snakemake.params.class_labels) > 2:
    cm = multilabel_confusion_matrix(Y_test, Y_pred, labels=y_classes)
    tn, fp, fn, tp = [i for i in sum(cm).ravel()]
else:
    cm = confusion_matrix(Y_test, Y_pred)
    tn, fp, fn, tp = cm.ravel()

accuracy = tp + tn / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# harmonic mean of precion and recall
F1 = 2*(precision * recall) / (precision + recall)
model_mcc = matthews_corrcoef(y_test, Y_pred)

with open(snakemake.log[0], "a") as logfile:
    printout = f"Model: LinearSVC | Precision: {precision:.4f} |\
        Recall: {recall:.4f} | MCC: {model_mcc:.4f} | \
            F1: {F1:.4f} | Accu: {accuracy:.4f}"
    logfile.write(printout)

# plot the 2D decision boundary
# from: https://github.com/suvoooo/Machine_Learning/blob/master/SVM_Decision_Boundary/Decision_Boundary_SVM.ipynb

# rerun feature selection, pca
X_test_df = pd.DataFrame(X_test,
                         columns=features_col)
ANOVA_filter = grid_cv.best_estimator_[0]
selector = grid_cv.best_estimator_[1]
ANOVA_selected = features_col[ANOVA_filter.get_support()]
lsvc_selected = ANOVA_selected[selector.get_support()]

X_test_selected = X_test_df[lsvc_selected].values

pca2 = PCA(n_components=2)
X_test_selected_reduced = pca2.fit_transform(X_test_selected)

classify = grid_cv.best_estimator_[3]


def plot_contours(ax, clf, xx, yy, **params):
    Z = classify.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([cdict2[z] for z in Z])
    print('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


X0, X1 = X_test_selected_reduced[:, 0], X_test_selected_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')

Y_tar_list = y_test.tolist()
labels1 = pd.Series(Y_tar_list)

full_labels = dict(zip(snakemake.params.class_labels,
                       snakemake.params.full_labels))

arr = [pd.Series(a) for a in [labels1, X0, X1]]
plot_df = pd.concat(arr, axis=1)
plot_df.columns = ["true_class", "PC_1", "PC_2"]
hue_order = snakemake.params.class_labels
cdict2 = dict(zip(hue_order, [1, 2, 3]))

# draw contours showing model predictions
cs = plot_contours(ax, classify, xx, yy,
                   colors=["limegreen", 'w', "deepskyblue", "w"],
                   alpha=0.5,
                   levels=len(snakemake.params.class_labels),
                   label="Prediction")

# scatter PC1, PC2 coords for each test instance
ax = sb.scatterplot(data=plot_df,
                    x="PC_1",
                    y="PC_2",
                    hue_order=hue_order,
                    hue="true_class",
                    palette=["limegreen",
                             "deepskyblue", "peru"][:len(hue_order)],
                    alpha=1,
                    s=60,
                    zorder=20,
                    ax=ax)

# highlight support vector instances
sv = ax.scatter(classify.support_vectors_[:, 0],
                classify.support_vectors_[:, 1],
                s=60, facecolors='none',
                edgecolors='k', label='Support Vectors',
                zorder=100)

# make proxy artists for contours colours
proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
         for pc in cs.collections if pc.get_facecolor()[0][0] != 1]

# custom legend
h, l = ax.get_legend_handles_labels()
l = [full_labels.get(e, e) for e in l]
h = h + [h[0]] + proxy
l = l + ["prediction"] + hue_order  # add subtitle for contours
plt.legend(h, l, fontsize=15, bbox_to_anchor=(1, 1.05))

xl = plt.xlabel("PC_1", fontsize=14)
yl = plt.ylabel("PC_2", fontsize=14)

# annotate w/t pipeline params
n_genes = lsvc_selected.shape[0]
classifier_params = grid_cv.best_estimator_[3].get_params()
kernel = classifier_params["kernel"]
gamma = classifier_params["gamma"]
C = classifier_params["C"]
accu = grid_cv_test_score
st = f"linearSVC n_genes={n_genes}, k={kernel}; γ={gamma}; C={C}; accuracy={accu:.3f} on PDx vsd expression"
st = fig.suptitle(st, y=.95, fontsize=18)
fig.tight_layout
fig.savefig(snakemake.output.boundary_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "expr_FeatCleanSelect"})

# pickle pipeline
model_filename = snakemake.output.featSelect_model
with open(model_filename, 'wb') as f:
    pickle.dump(grid_cv.best_estimator_, f)

svm = grid_cv.best_estimator_[3]
selector = grid_cv.best_estimator_[1]

# get selected feature
input_matrix = merge_matrix
input_matrix.index = input_matrix.index.values

TT_df = drug_response_data[drug_response_data.ircc_id.isin(input_matrix.index)][
    ["ircc_id", "is_test"]]
train_models = TT_df[TT_df.is_test == False].ircc_id.unique()
test_models = TT_df[TT_df.is_test == True].ircc_id.unique()

features_col = np.array([c for c in input_matrix.columns if c != target_col])

# slice selected features, save train test
X_train = input_matrix.loc[train_models, lsvc_selected]
y_train = input_matrix.loc[train_models, target_col]
X_test = input_matrix.loc[test_models, lsvc_selected]
y_test = input_matrix.loc[test_models, target_col]

# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                       columns=lsvc_selected,
                       index=X_train.index)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                      columns=lsvc_selected,
                      index=X_test.index)
X_train.to_csv(snakemake.output.Xtrain, sep="\t")
X_test.to_csv(snakemake.output.Xtest, sep="\t")

if len(svm.classes_) > 2:
    # get the feature selector (linear SVC) coeff
    coeff_plot_df = pd.DataFrame(selector.estimator_.coef_.T,
                                 columns=svm.classes_,
                                 index=ANOVA_selected)
    # keep only supported features
    coeff_plot_df["support"] = selector.get_support()
    coeff_plot_df = coeff_plot_df[coeff_plot_df["support"]
                                  == True][svm.classes_]
    coeff_plot_df = coeff_plot_df.stack().reset_index()
    coeff_plot_df.columns = ["feature", "class", "coeff"]
    coeff_plot_df = coeff_plot_df.sort_values("coeff")

    # select top / bottom features
    top = pd.concat(
        [coeff_plot_df.head(10), coeff_plot_df.tail(10)]).feature.unique()
    plot_df = coeff_plot_df[coeff_plot_df.feature.isin(top)]

    fig, ax = plt.subplots(figsize=(10, 16))
    ax = sb.barplot(x="coeff",
                    y="feature",
                    hue="class",
                    hue_order=sorted(y_classes),
                    palette="Set2",
                    data=plot_df)
else:
    # get linear SVC feature coefficients
    coeff_plot_df = pd.DataFrame(selector.estimator_.coef_.T,
                                 index=ANOVA_selected)
    # keep only supported features
    coeff_plot_df["support"] = selector.get_support()
    coeff_plot_df = coeff_plot_df[coeff_plot_df["support"] == True]
    coeff_plot_df = coeff_plot_df.stack().reset_index()
    coeff_plot_df.columns = ["feature", "class", "coeff"]
    coeff_plot_df = coeff_plot_df.sort_values("coeff")
    print(coeff_plot_df.head())

    # select top / bottom features
    top = pd.concat(
        [coeff_plot_df.head(10), coeff_plot_df.tail(10)]).feature.unique()
    plot_df = coeff_plot_df[coeff_plot_df.feature.isin(top)]

    fig, ax = plt.subplots(figsize=(10, 16))
    ax = sb.barplot(x="coeff",
                    y="feature",
                    palette="Set2",
                    data=plot_df)
st = fig.suptitle(printout, y=.95, fontsize=18)
fig.tight_layout
fig.savefig(snakemake.output.loadings_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "expr_FeatCleanSelect"})
