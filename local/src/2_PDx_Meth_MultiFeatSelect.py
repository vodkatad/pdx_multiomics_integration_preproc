#!/usr/bin/env python
# coding: utf-8

# Data manipulation
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import multilabel_confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import vaex

# Options for pandas
# No warnings about setting value on copy of slice
pd.options.mode.chained_assignment = None

# scikit-learn
# scalers
# processing
# models
# feature selection
# benchmark
# Hyperparameter tuning

# Visualizations
# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)

# read methylation M file in chunks, convert to to hd5
# in_filename="data/methylation/m_values_Umberto.tsv"
#master_filename = 'data/methylation/m_values_Umberto.hd5'
#numerical_cols = pd.read_csv("data/methylation/columns.tsv", header=None)[0][1:].tolist()
#chunksize = 10000
#chunk_list = []
#i = 0
# for chunk in pd.read_csv(in_filename,
#                sep="\t",
#                iterator=True,
#                chunksize=chunksize,
#                names=["probe"] + numerical_cols,
#                header=None,
#                skiprows=1):
#        # covert to np float
#        chunk[numerical_cols] = chunk[numerical_cols].apply(pd.to_numeric,
#                                                            errors='coerce',
#                                                            downcast='float')
#        chunk["probe"] = chunk["probe"].astype(str)
#        vaex_df = vaex.from_pandas(chunk, copy_index=False)
#        out_filename = f"data/methylation/m_values/m_values_{i}.hd5"
#        vaex_df.export_hdf5(path=out_filename)
#        i+=1
#        chunk_list.append(out_filename)
#        del chunk
#master_df = vaex.open_many(chunk_list)
# master_df.export_hdf5(path=master_filename)

master_filename = snakemake.input.meth_m
features_in = vaex.open(master_filename)
nrows, ncols = features_in.shape

# read methylation bionomial test pvalue
bDTpval = pd.read_csv(snakemake.input.meth_B_DTpval,
                      sep="\t").reset_index()
bDTpval.columns = ["probe", "beta_DT-pvalue"]
bDTpval = bDTpval.set_index("probe")
# read methylation M variance
Msd = pd.read_csv(snakemake.input.meth_m_sd,
                  sep="\t").reset_index()
Msd.columns = ["probe", "M_sd"]
Msd = Msd.set_index("probe")
probe_stats = pd.concat([Msd, bDTpval], axis=1)

# keep probes w/t binomial FDR < .05,
# keep most variable probes
sd_pctl = snakemake.params.sd_pctl
M_sd_thrs = probe_stats.describe(percentiles=[sd_pctl[:-1]/100]).\
    loc[sd_pctl, "M_sd"]
FDR = .05
probes_tokeep = pd.Series(probe_stats[(
    probe_stats["beta_DT-pvalue"] < FDR) & (probe_stats["M_sd"] > M_sd_thrs)].index.tolist())

# load sample id conversion table, drug response data
drug_response_data = pd.read_csv(snakemake.input.response,
                                 sep="\t")

# filter on probes, models
samples_tokeep = drug_response_data.ircc_id.apply(
    lambda x: x.replace("TUM", " ").split()[0]).unique()
samples_tokeep = [c for c in samples_tokeep if c in list(
    features_in.columns.keys())[1:]]
features_clean = features_in[features_in["probe"].isin(
    probes_tokeep)][samples_tokeep + ["probe"]]

# convert to pandas, reshape, add target
target_col = snakemake.params.target_col
features_clean_df = features_clean.to_pandas_df()
features_clean_df = features_clean_df.set_index("probe").T
features_clean_df.columns = features_clean_df.columns.tolist()
features_clean_df["ircc_id_short"] = [x[0:7] for x in features_clean_df.index]
features_clean_df = pd.merge(drug_response_data[[
    target_col, "ircc_id_short", "ircc_id", "is_test"]],
    features_clean_df,
    on="ircc_id_short")

train_models = features_clean_df[features_clean_df.is_test ==
                                 False].ircc_id.unique()
test_models = features_clean_df[features_clean_df.is_test ==
                                True].ircc_id.unique()
features_clean_df = features_clean_df.drop(
    ["is_test", "ircc_id_short"], axis=1).set_index("ircc_id")

input_matrix = features_clean_df
input_matrix.index = input_matrix.index.values
features_col = np.array([c for c in input_matrix.columns if c != target_col])

# train-test split
X_train = input_matrix.loc[train_models, features_col].values
y_train = input_matrix.loc[train_models, target_col].values
X_test = input_matrix.loc[test_models, features_col].values
y_test = input_matrix.loc[test_models, target_col].values

# standardise features
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# train the feature selector inside a pipeline that tries to maximise
# classification accuracy on the training set
N = len(features_col)
Ks = [int(f) for f in [N/100, N/50, N/10, N/5]]
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

# Set up the random search with 4-fold cross validation
grid_cv = GridSearchCV(estimator=pipeline,
                       param_grid=hyperparameter_grid,
                       cv=4,  # n_iter=100,
                       scoring="accuracy",
                       n_jobs=-1, refit=True,
                       return_train_score=True)


CVresult_df = pd.DataFrame(grid_cv.cv_results_)
CVresult_df.sort_values("rank_test_score")[
    ["rank_test_score", "mean_train_score", "mean_test_score"]].head()

grid_cv_test_score = grid_cv.score(X_test, y_test)

y_classes = input_matrix[target_col].unique().tolist()
Y_pred = grid_cv.predict(X_test)
# print (Y_pred)
multi_cm = multilabel_confusion_matrix(y_test, Y_pred, labels=y_classes)
tn, fp, fn, tp = [i for i in sum(multi_cm).ravel()]
accuracy = tp + tn / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
# harmonic mean of precion and recall
F1 = 2*(precision * recall) / (precision + recall)
model_mcc = matthews_corrcoef(y_test, Y_pred)
printout = f"{grid_cv.best_estimator_} \n Precision: {precision:.4f} |Recall: {recall:.4f}  |MCC: {model_mcc:.4f}  |F1: {F1:.4f} |Accu: {accuracy:.4f}"

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

full_labels = {"OR": "Objective Response (<-50%)",
               "PD": "Progressive Disease (>35%)",
               "SD": "Stable Disease"}


arr = [pd.Series(a) for a in [labels1, X0, X1]]
plot_df = pd.concat(arr, axis=1)
plot_df.columns = ["true_class", "PC_1", "PC_2"]
hue_order = ["OR", "PD", "SD"]
cdict2 = dict(zip(hue_order, [1, 2, 3]))

# draw contours showing model predictions
cs = plot_contours(ax, classify, xx, yy,
                   colors=["limegreen", "peru", "w", "deepskyblue", "w"],
                   alpha=0.5, levels=3, label="Prediction")

# scatter PC1, PC2 coords for each test instance
ax = sb.scatterplot(data=plot_df,
                    x="PC_1",
                    y="PC_2",
                    hue_order=hue_order,
                    hue="true_class",
                    palette=["limegreen", "peru", "deepskyblue"],
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
classifier_params = grid_cv.best_estimator_[2].get_params()
kernel = classifier_params["kernel"]
gamma = classifier_params["gamma"]
C = classifier_params["C"]
accu = grid_cv_test_score
st = f"SVC n_genes={n_genes}, k={kernel}; γ={gamma}; C={C}; accuracy={accu:.3f} on PDx meth M"
st = fig.suptitle(st, y=.95, fontsize=18)

fig.savefig(snakemake.output.boundary_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "meth_FeatCleanSelect"})

# pickle pipeline
model_filename = snakemake.output.featSelect_model
with open(model_filename, 'wb') as f:
    pickle.dump(grid_cv.best_estimator_, f)

svm = grid_cv.best_estimator_[3]
selector = grid_cv.best_estimator_[1]
# get selected feature
input_matrix = features_clean_df
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
# standardise features
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train.values),
                       columns=lsvc_selected,
                       index=X_train.index)
X_test = pd.DataFrame(StandardScaler().fit_transform(X_test.values),
                      columns=lsvc_selected,
                      index=X_test.index)
X_train.to_csv(snakemake.output.Xtrain, sep="\t")
X_test.to_csv(snakemake.output.Xtest, sep="\t")

# get the feature selector (linear SVC) coeff
coeff_plot_df = pd.DataFrame(selector.estimator_.coef_.T,
                             columns=svm.classes_,
                             index=ANOVA_selected)
# keep only supported features
coeff_plot_df["support"] = selector.get_support()
coeff_plot_df = coeff_plot_df[coeff_plot_df["support"] == True][svm.classes_]
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
                palette="Set2",
                data=plot_df)
st = fig.suptitle(printout, y=.95, fontsize=18)
fig.tight_layout
fig.savefig(snakemake.output.loadings_fig,
            format='pdf',
            bbox_inches='tight',
            dpi=fig.dpi,
            metadata={"Creator": "expr_FeatCleanSelect"})
