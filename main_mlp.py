#%%
import numpy as np
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from functools import reduce
import sys
import time

#%%
M = loadmat('T4/xdata.mat')

Xtest = M['Xtest']
Xtrain = M['Xtrain']
Xval = M['Xval']
ytest = M['ytest'].ravel()
ytrain = M['ytrain']
yval = M['yval']

Xfit = np.concatenate((Xtrain, Xval))
yfit = np.concatenate((ytrain, yval)).ravel()

#%%

layers = [(50,50,50), (50,100,50), (8,8,8), (8,10,8)]
layers_names = ['(50,50,50)', '(50,100,50)', '(8,8,8)', '(8,10,8)']
activations = ['tanh', 'relu']
solvers = ['sgd', 'adam']
alphas = [0.0001, 0.05]
lrs = ['constant','adaptive']

tracking_val = {}
for layer in layers_names:
    tracking_val[layer] = {}
    for activation in activations:
        tracking_val[layer][activation] = {}
        for solver in solvers:
            tracking_val[layer][activation][solver] = {}
            for alpha in alphas:
                tracking_val[layer][activation][solver][alpha] = {}
                for lr in lrs:
                    tracking_val[layer][activation][solver][alpha][lr] = None

parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (8,8,8), (8,10,8)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

#%%

startTime = time.time()
cv = [(slice(None), slice(None))] # Hack para no hacer cv
mlp = MLPClassifier(early_stopping=True, validation_fraction=5.0/6)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=cv)
clf.fit(Xfit, yfit)
print("--- %s seconds ---" % (time.time() - startTime))

#%%
accs_val = clf.cv_results_['mean_test_score']
for acc_val, params in zip(accs_val, clf.cv_results_['params']):
    layer = layers_names[layers.index(params['hidden_layer_sizes'])]
    activation, solver, alpha, lr = [params[par] for par in ['activation', 'solver', 'alpha', 'learning_rate']]
    tracking_val[layer][activation][solver][alpha][lr] = acc_val

tracking_val['highscore'] = clf.best_score_
tracking_val['hyperparams'] = clf.best_params_
tracking_val['acc'] = clf.score(Xtest, ytest)

import json
with open('tracking_val_mlp.json', 'w') as outfile:
    json.dump(tracking_val, outfile)

#%%
clf.best_params_