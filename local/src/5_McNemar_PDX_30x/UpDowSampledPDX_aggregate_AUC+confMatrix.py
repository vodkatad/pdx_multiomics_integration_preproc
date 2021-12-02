
import seaborn as sb
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import distance
import re

from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score, accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
# Hyperparameter tuning

# Visualizations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Set default font size
plt.rcParams['font.size'] = 24
# Set default font size
sb.set(font_scale=.8)
custom_style = {'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black'}
sb.set_style("white", rc=custom_style)

target_col = snakemake.params.target_col
Y_class_dict = {0: 'PD', 1: 'SD-OR'}


def evaluate_model(modelFile, XtestFile, YtestFile, model_name):
    # load the model from file
    classifier = pickle.load(open(modelFile, 'rb'))
    # load test set
    X_test = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
    y_test = pd.read_csv(YtestFile, sep="\t", header=0,
                         index_col=0)[target_col]
    # make sure test, train have same instances
    X_test = X_test.loc[y_test.index.tolist()]
    # assess best classifier performance on test set (accuracy)
    grid_test_score = classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    # convert binary target variable to readable labels
    y_test_readable = [Y_class_dict[y] for y in y_test]
    y_pred_readable = [Y_class_dict[y] for y in y_pred]

    # compute and flatten confusion matrix
    confMatrix = confusion_matrix(y_test_readable,
                                  y_pred_readable,
                                  labels=['PD', 'SD-OR'])
    confusion_df = pd.DataFrame(confMatrix,
                                columns=['pred_PD', 'pred_SD-OR'],
                                index=['true_PD', 'true_SD-OR']).stack()
    confusion_df.index = ['__'.join(ix) for ix in confusion_df.index]
    # compute the "marginal probability" according to the model
    # that the given instance has the predicted label
    y_test_predict_proba = classifier.predict_proba(X_test)
    AUC = roc_auc_score(y_test, y_test_predict_proba[:, 1])
    return [model_name,
            grid_test_score,
            AUC] + confusion_df.tolist()

tab_arr = []
all_models = snakemake.input.up_models + snakemake.input.down_models 
for stackedFile in all_models:
    # extract holdout shuffle index from filename
    split_index = int(float(
        re.findall(r'\d+\.', stackedFile.split("_")[-1])[0]))
    # get the corresponding test set
    XtestFile_eng = snakemake.input.X_eng[split_index]
    YtestFile = snakemake.input.Y[split_index] 
    # extract down / up samapling percent from filenname
    trainSet_label = stackedFile.split("_")[0].\
        replace('Cetuximab', '').replace('sampled', '_')
    # evaluate model performance on test set
    tab_arr.append([split_index,  trainSet_label] + \
        evaluate_model(stackedFile,
        XtestFile_eng, YtestFile, 'PDXstackedCVClassifier'))
    split_index += 1
out_tab = pd.DataFrame(tab_arr, columns=["split_index", 
                                        "trainSet_label",
                                        'model_name',
                                         "grid_test_accu",
                                         "AUC"] + ['true_PD__pred_PD',
                                                   'true_PD__pred_SD-OR',
                                                   'true_SD-OR__pred_PD',
                                                   'true_SD-OR__pred_SD-OR'])
out_tab.to_csv(snakemake.output.performance_tab, sep='\t')
