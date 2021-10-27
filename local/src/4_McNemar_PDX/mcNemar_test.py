# https://en.wikipedia.org/wiki/McNemar%27s_test
# http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/#example-2-mcnemars-test-for-scenario-b
# McNemar's Test [McNemar, Quinn, 1947. "Note on the sampling error of the difference between correlated proportions or percentages". Psychometrika. 12 (2): 153–157.]
#  (sometimes also called "within-subjects chi-squared test") is a statistical 
# test for paired nominal data. In context of machine learning (or statistical) models, 
# we can use McNemar's Test to compare the predictive accuracy of two models. 
# McNemar's test is based on a 2 times 2 contigency table of the two model's predictions.

import numpy as np
import pickle
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table

def McNemar_test(y_target, y_model1, y_model2):
	contingency_tab = mcnemar_table(y_target=y_target, 
			y_model1=y_model1, 
			y_model2=y_model2)
	chi2, p = mcnemar(ary=contingency_tab, exact=True)
	return chi2, p

def evaluate_model(modelfile, y_target):
	## load the model from file
#classifier = pickle.load(open(model_filename, 'rb'))
## assess best classifier performance on test set
#grid_test_score = classifier.score(X_test, y_test)
#y_pred = classifier.predict(X_test)
#print(f'Accuracy on test set: {grid_test_score:.3f}')

## print classification report on test set
#print(classification_report(y_test, y_pred, target_names=['PD', 'SD-OR']))

##confusion_matrix = confusion_matrix(y_test, y_pred)
#plot_confusion_matrix(classifier, X_test, y_test,
                                 #display_labels=['PD', 'SD-OR'],
                                 #cmap=plt.cm.Blues)

## return the marginal probability that the given sample has the label in question
#y_test_predict_proba = classifier.predict_proba(X_test)
#roc_auc_score(y_test, y_test_predict_proba[:, 1])

# load drug response data
f = snakemake.input.response
Y = pd.read_csv(f, sep="\t", index_col=1, header=0)
# encode target var (binary responder/non-responder)
target_col = snakemake.params.target_col
Y_class_dict={'PD':0,'OR+SD':1}
y_target = Y[Y.is_test == True].\
	target_col.replace(Y_class_dict).values

for duo in zip(snakemake.input.model_set1, snakemake.input.model_set1):
	modeFile1, modelFile2 = duo

