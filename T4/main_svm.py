#%% Primero importamos las librerías que usaremos 
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from functools import reduce
import sys
import time

#%% Definimos un proceso de visualización que indica qué porcentaje del espacio de parámetros
# ha sido recorrido
def progressBar(value, endvalue, current, currentKernel, currentGamma, currentC, currentDegree,
kernel, gamma, c, degree, bar_length=20):
    
    # Visaulización obtenida de https://stackoverflow.com/questions/6169217/replace-console-output-in-python
    
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rCurrent: {2} params: {3},{6},{7},{8} Progress : [{0}] {1}% = {4}/{5} training: {9},{10},{11},{12}".format(arrow + spaces, int(round(percent * 100)), current, kernel, value, endvalue, gamma, c, degree, currentKernel, currentGamma, currentC, currentDegree))
    sys.stdout.flush()

#%% Creamos variables para los datos que usaremos.
filepath  = 'xdata.mat'
M = loadmat(filepath)

Xtest = M['Xtest']
Xtrain = M['Xtrain']
Xval = M['Xval']
ytest = M['ytest'].ravel()
ytrain = M['ytrain'].ravel()
yval = M['yval'].ravel()

#%% Creamos el diccionario tracking_val, donde se rastreará el resultado
# de cada iteración para determinar los mejores hiperparámetros. Además se
# genera el espacio de los parámetros, en donde se especifica en qué valores
# iterar
kernels = ['rbf', 'sigmoid', 'poly']
gammas = [2**(i) for i in range(-8, -3, 2)]
gammas.append('auto')
cs = [2**(i) for i in range(-2, 6, 2)]
degrees = [i for i in range(4)]

tracking_val = {}
for kernel in kernels:
    tracking_val[kernel] = {}
    for gamma in gammas:
        tracking_val[kernel][gamma] = {}
        for c in cs:
            tracking_val[kernel][gamma][c] = {}
            for degree in degrees:
                tracking_val[kernel][gamma][c][degree] = None

highscore = None
hyperparametersHighscore = None

number_of_cases = 1
for l in [kernels, gammas, cs, degrees]:
    number_of_cases *= len(l)
#%% Se realiza el entrenamiento y medimiento en validación que cada modelo SVM con
# sus modificaciones
startTime = time.time()
i = 0
for kernel in kernels:
    for gamma in gammas:
        for c in cs:
            for degree in degrees:
                svmClassifier = SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)
                svmClassifier.fit(Xtrain, ytrain)
                acc = svmClassifier.score(Xval, yval)

                if highscore == None or acc > highscore:
                    highscore = acc
                    hyperparametersHighscore = [kernel, gamma, c, degree]

                if kernel != 'poly':
                    for d in degrees:
                        tracking_val[kernel][gamma][c][d] = acc
                    i += len(degrees)
                    progressBar(i, number_of_cases, highscore, kernel, gamma, c, degree, *hyperparametersHighscore)
                    break
                
                tracking_val[kernel][gamma][c][degree] = acc
                i += 1
                progressBar(i, number_of_cases, highscore, kernel, gamma, c, degree, *hyperparametersHighscore)

print("--- %s seconds ---" % (time.time() - startTime))
                

#%% Dado el mejor modelo, se obtiene el accuaracy de este en el set de test

kernel, gamma, c, degree = hyperparametersHighscore
svmClassifier =  SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)
svmClassifier.fit(Xtrain, ytrain)
acc = svmClassifier.score(Xtest, ytest)

#%% Se guardan los resultados en tracking_val
tracking_val['highscore'] = highscore
tracking_val['hyperparams'] = hyperparametersHighscore
tracking_val['acc'] = acc

import json
with open('tracking_val.json', 'w') as outfile:
    json.dump(tracking_val, outfile)