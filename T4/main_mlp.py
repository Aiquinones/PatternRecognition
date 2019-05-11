#%% Primero importamos las librerías que usaremos 
import numpy as np
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from functools import reduce
import sys
import time

#%% Creamos variables para los datos que usaremos. Bajo la
# autorización del profesor, se permitió mezclar las datasets
# de train y validation, dividiéndolo luego en la misma proporción.
#  Esto para que sea aceptado como input al GridSearchCV
#  
filepath  = 'xdata.mat'
M = loadmat(filepath)

Xtest = M['Xtest']
Xtrain = M['Xtrain']
Xval = M['Xval']
ytest = M['ytest'].ravel()
ytrain = M['ytrain']
yval = M['yval']

Xfit = np.concatenate((Xtrain, Xval))
yfit = np.concatenate((ytrain, yval)).ravel()

#%% Creamos el diccionario tracking_val, donde se rastreará el resultado
# de cada iteración para determinar los mejores hiperparámetros. Además se
# genera el espacio de los parámetro, en donde se especifica en qué valores
# iterar

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

#%% Se itera el entrenamiento del modelo para obtener la mejor combinación de hiperparámetros

startTime = time.time()
cv = [(slice(None), slice(None))] # Hack para no hacer cv
mlp = MLPClassifier(early_stopping=True, validation_fraction=5.0/6)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=cv)
clf.fit(Xfit, yfit)
print("--- %s seconds ---" % (time.time() - startTime))

#%% Se guardan los resultados en tracking_val_mlp
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