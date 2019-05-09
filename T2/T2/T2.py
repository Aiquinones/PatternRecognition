import json
from pybalu.feature_selection import clean, sfs
from pybalu.feature_extraction import lbp_features
from sklearn.neighbors import KNeighborsClassifier
from pybalu.performance_eval import performance
from pybalu.io import imread
import numpy as np
import os
import sys

guardar = False
ejemplo = False


def progressBar(value, endvalue, bar_length=20):
    '''
    Visaulización obtenida de
    https://stackoverflow.com/questions/6169217/replace-console-output-in-python
    '''

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces,
                                                    int(round(percent * 100))))
    sys.stdout.flush()


# # Parte 1

# 1) Elegimos hdiv = 2 y vdiv = 2 dado que entregan buenos resultados.

hdiv, vdiv = 2, 2

directory = 'faces_ARLQ'
LBPs = []
IDs = []
NNs = []

i = 0
for filename in os.listdir(directory):

    if filename.endswith(".png"):

        # Para un archivo "face_xxx_nn.png", ID es xxx y nn es nn, ambos en int
        id_nn = filename[:-4].split("_")
        ID = int(id_nn[1])
        nn = int(id_nn[2])

        if ID % 2 == 1:  # Nos quedamos con los impares
            if nn <= 7:  # y solo los 7 primeros
                i += 1

                # Leemos la imágen y obtenemos sus features dadas por lbp
                im = imread(f"{directory}/{filename}")
                lbp = lbp_features(im, hdiv=hdiv, vdiv=vdiv)

                # Guardamos los resultados
                LBPs.append(lbp)
                IDs.append(ID)
                NNs.append(nn)

                if i % 10 == 0:
                    progressBar(i, 350, bar_length=20)
print(f"\nLBPs Calculado\nLBP Shape: {len(LBPs)}, {len(LBPs[0])}\nIDs" +
      f"Shape: {len(IDs)}, 1")


# 2) Antes de la selección de features, realizaremos un cleansing con Clean.
# Luego realizaremos SFS.


Xtrain, Xtest, Ytrain, Ytest = [], [], [], []

for lbp, ID, nn in zip(LBPs, IDs, NNs):

    # Como nos pide el enunciado, usamos las imágenes con nn=1 como testing,
    # y el resto como training

    if nn == 1:
        Xtest.append(lbp)
        Ytest.append(ID)
    else:
        Xtrain.append(lbp)
        Ytrain.append(ID)

# Transformamos a numpy arrays para que sea más fácil de trabajar (por pybalu)
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)


before = len(Xtrain[0])

# Realizamos la función clean previo a la selection de features mediante sfs,
# como se nos fue recomendados en clase
p_clean = clean(Xtrain)

after = len(p_clean)
print(f"Cleaned.\nBefore:{before} features\nNow: {after} features")


# Después de obtener qué features usaremos, modificamos nuestro dataset a solo
# contener esas features
Xtrain_cleaned = np.array([[x[i] for i in p_clean] for x in Xtrain])
Xtest_cleaned = np.array([[x[i] for i in p_clean] for x in Xtest])


# Un bug en el código de sfs no permite que los las clases se salten
# enteros (1,3,5,..). Modificamos esto solo para correr sfs
Ytrain_sfs = np.array([int((y-1)/2) for y in Ytrain])

# Obtenemos qué features sfs nos ha seleccionado
p_sfs = sfs(Xtrain_cleaned, Ytrain_sfs, 100, show=True)


# Nuevamente modificamos nuestro dataset para solo considerar las features
# dadas por el selector
Xtrain_sfs = np.array([[x[i] for i in p_sfs] for x in Xtrain_cleaned])
Xtest_sfs = np.array([[x[i] for i in p_sfs] for x in Xtest_cleaned])


# Inicializamos el clasificador con k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xtrain_sfs, Ytrain)


# 3) Calculamos el accuracy usando la función performance entregada en pybalu


# Predecimos y vemos el resultado
pred = knn.predict(Xtest_sfs)
print(f"Predicción tuvo un accuracy de {performance(pred, Ytest)}")


# # Parte 2

# 1) Al igual que en la parte 1, usamos un $hdiv = 2$ y $vdiv = 2$

LBPs2 = []
IDs2 = []
NNs2 = []

i = 0
for filename in os.listdir(directory):

    if filename.endswith(".png"):

        # Para un archivo "face_xxx_nn.png", ID es xxx y nn es nn, ambos en int
        id_nn = filename[:-4].split("_")
        ID = int(id_nn[1])
        nn = int(id_nn[2])

        if ID % 2 == 0:  # Nos quedamos con los pares
            if nn <= 7:  # y solo los 7 primeros
                i += 1

                # Leemos la imágen y obtenemos sus features dadas por lbp
                im = imread(f"{directory}/{filename}")
                lbp = lbp_features(im, hdiv=hdiv, vdiv=vdiv)

                # Guardamos los resultados
                LBPs2.append(lbp)
                IDs2.append(ID)
                NNs2.append(nn)

                if i % 10 == 0:
                    progressBar(i, 350, bar_length=20)
print(f"\nLBPs Calculado\nLBP Shape: {len(LBPs)}, {len(LBPs[0])}\nIDs Shape:" +
      f"{len(IDs)}, 1")


# 2)

Xtrain2, Xtest2, Ytrain2, Ytest2 = [], [], [], []

# Como nos pide el enunciado, usamos las imágenes con nn=1 como testing, y el
# resto como training
for lbp, ID, nn in zip(LBPs2, IDs2, NNs2):
    if nn == 1:
        Xtest2.append(lbp)
        Ytest2.append(ID)
    else:
        Xtrain2.append(lbp)
        Ytrain2.append(ID)

# Transformamos a numpy arrays para que sea más fácil de trabajar (por pybalu)
Xtrain2 = np.array(Xtrain2)
Ytrain2 = np.array(Ytrain2)
Xtest2 = np.array(Xtest2)
Ytest2 = np.array(Ytest2)


# Para obtener las mismas features del clean de la parte 1, usamos el p_clean
# (en vez de calcular uno nuevo)
Xtrain2_cleaned = np.array([[x[i] for i in p_clean] for x in Xtrain2])
Xtest2_cleaned = np.array([[x[i] for i in p_clean] for x in Xtest2])


# Así, usamos p_sfs de la parte 1 para obtener las mismas features
Xtrain2_sfs = np.array([[x[i] for i in p_sfs] for x in Xtrain2_cleaned])
Xtest2_sfs = np.array([[x[i] for i in p_sfs] for x in Xtest2_cleaned])


# 3)

# Inicializamos otra vez el clasificador con k=1
knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(Xtrain2_sfs, Ytrain2)

# 4)

# Nuevamente predecimos y vemos el resultado
pred2 = knn2.predict(Xtest2_sfs)
print(f"Predicción tuvo un accuracy de {performance(pred2, Ytest2)}")


# # Guardar resultados

# Los ndarray nos son JSON serializable, por lo que los pasamos a una lista de
# ints
p_clean_l = [int(p) for p in p_clean]
p_sfs_l = [int(p) for p in p_sfs]

# Guardamos lo necesario para que el proceso no se tenga que ejecutar
# nuevamente

if guardar:
    data = {}
    data['hdiv'] = hdiv
    data['vdiv'] = vdiv
    data['p_clean'] = p_clean_l
    data['p_sfs'] = p_sfs_l

    with open('saved.txt', 'w') as out:
        json.dump(data, out)

# En caso de querer correr el modelo nuevamente, se usaría esta función.
# Notamos que knn es un algoritmo 'lazy', por lo que requiere el Xtrain y
# Ytrain. Si se prefiere pasar el modelo ya fiteado se da la opción


def predict_from_scratch(filepath, data_filename, Xtrain=None, Ytrain=None,
                         knn=None):

    if Xtrain is None or Ytrain is None:
        assert knn, "Por favor dar el modelo, o bien los datos Xtrain y Ytrain"

    # Obtenemos los valores guardados
    with open(data_filename) as f:
        data = json.load(f)

    hdiv = int(data['hdiv'])
    vdiv = int(data['vdiv'])
    local_p_clean = np.array(data['p_clean'])
    local_p_sfs = np.array(data['p_sfs'])

    # Calculamos las features
    im = imread(filepath)
    lbp = lbp_features(im, hdiv=hdiv, vdiv=vdiv)

    # Filtramos
    lbp_clean = np.array([lbp[i] for i in local_p_clean])
    lbp_sfs = np.array([lbp_clean[i] for i in local_p_sfs])
    lbp_sfs = lbp_sfs.reshape(1, -1)

    if not knn:
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(Xtrain, Ytrain)

    local_pred = knn.predict(lbp_sfs)
    print(f"La predicción es: {local_pred}")


# Ejemplo
if ejemplo:
    predict_from_scratch('faces_ARLQ/face_001_03.png', 'saved.txt', knn=knn)
