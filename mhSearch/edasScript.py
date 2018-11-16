import sys
sys.path.append("./lib/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from sklearn.model_selection import GridSearchCV
from lib.Methods import GeneralMethods
from lib.edasSearch import EdasHyperparameterSearch
from lib.Hiperparametros import HyperparameterSwitcher
from lib.ImportacionModelos import getClassifierNames
from lib.ImportacionModelos import getClassifierModels
from lib.ImportacionModelos import getRegressorNames
from lib.ImportacionModelos import getRegressorModels
#from lib.graphicGenerator import GraphicBuilder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

seed = 9
xSize = 1041
kf = KFold(n_splits=10)
df = pd.read_csv("data/filtred.csv")
X = df[df.columns[:xSize]]
Y = df[df.columns[xSize:]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
#gbTrain = GraphicBuilder(pd.concat([X_train, y_train],axis=1))
#gbTest = GraphicBuilder(pd.concat([X_test, y_test],axis=1))


estimadorDictionary = getClassifierModels(includeEnsambled=True)
hypSwitcher = HyperparameterSwitcher()
process = 'edas'
for idModeloPrueba in range(len(estimadorDictionary)):
    result = {}
    modelName = getClassifierNames(includeEnsambled=True)[idModeloPrueba]
    estimador = estimadorDictionary[modelName]
    parametros = hypSwitcher.getHyperparameters(modelName)()
    try:
        ss = reduce(lambda x,y: x*y, map(lambda x: len(parametros[x]), parametros.keys()))
        if ss>10:
            iteraciones=4
            muestra=5
        else:
            iteraciones=8
            muestra=int(np.ceil(ss*0.5))
        gm = GeneralMethods(estimador, X_train, y_train.FLOOR, seed=seed)
        test = EdasHyperparameterSearch(gm, parametros, estimador, 
                iterations=iteraciones, sample_size=muestra, select_ratio=0.5, debug=False) # sample_size*select_ratio>=1
        test.run()
        dff = pd.DataFrame(list(test.resultados)).sort_values(['Accuracy'], ascending=False).reset_index(drop=True)
        dff.to_csv("result/" + process + "/" + modelName + "floor.csv", index=False)
    except:
        dff = pd.DataFrame()
        dff.to_csv("result/" + process + "/" + modelName + "-floor-FALLO.csv", index=False)

for idModeloPrueba in range(len(estimadorDictionary)):
    result = {}
    modelName = getClassifierNames(includeEnsambled=True)[idModeloPrueba]
    estimador = estimadorDictionary[modelName]
    parametros = hypSwitcher.getHyperparameters(modelName)()
    try:
        ss = reduce(lambda x,y: x*y, map(lambda x: len(parametros[x]), parametros.keys()))
        if ss>10:
            iteraciones=4
            muestra=5
        else:
            iteraciones=8
            muestra=int(np.ceil(ss*0.2))
        gm = GeneralMethods(estimador, X_train, y_train.BUILDINGID, seed=seed)
        test = EdasHyperparameterSearch(gm, parametros, estimador, 
                iterations=iteraciones, sample_size=muestra, select_ratio=0.5, debug=False) # sample_size*select_ratio>=1
        test.run()
        dff = pd.DataFrame(list(test.resultados)).sort_values(['Accuracy'], ascending=False).reset_index(drop=True)
        dff.to_csv("result/" + process + "/" + modelName + "building.csv", index=False)
    except:
        dff = pd.DataFrame()
        dff.to_csv("result/" + process + "/" + modelName + "-building-FALLO.csv", index=False)
