
import seaborn as sb
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import distance

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

# load triple negative feature
tripleNeg_df = pd.read_csv(snakemake.input.tripleNeg,
                           sep='\t', header=0, index_col=0)['KRAS_BRAF_NRAS_triple_neg']
target_col = snakemake.params.target_col
Y_class_dict = {0: 'PD', 1: 'SD-OR'}


def get_rates(modelFile, XtestFile, YtestFile, model_name):
    # load the model from file
    classifier = pickle.load(open(modelFile, 'rb'))
    # load test set
    X_test = pd.read_csv(XtestFile, sep="\t", header=0, index_col=0)
    y_test = pd.read_csv(YtestFile, sep="\t", header=0,
                         index_col=0)[target_col]
    y_test_predict_proba = classifier.predict_proba(X_test)
    fp_rates, tp_rates, _ = roc_curve(y_test,y_test_predict_proba[:,1])


tab_arr = []
split_index = 0
for tupla in zip(snakemake.input.stacked_models,
                 snakemake.input.X_eng,
                 snakemake.input.Y):
    