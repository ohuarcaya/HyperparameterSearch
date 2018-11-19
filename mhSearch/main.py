import sys
import imp
import numpy as np
import pandas as pd
from functools import reduce
from multiprocessing import cpu_count
sys.path.append("./lib/")
from sklearn.model_selection import train_test_split
from lib.Hiperparametros import HyperparameterSwitcher
from lib.ImportacionModelos import getClassifierNames
from lib.ImportacionModelos import getClassifierModels
from lib.ImportacionModelos import getRegressorNames
from lib.ImportacionModelos import getRegressorModels
from lib.ProcessManager import Evaluator

"""
Ingesta de Datos
"""
seed = 9
xSize = 1041
df = pd.read_csv("data/filtred.csv")
X = df[df.columns[:xSize]]
Y = df[df.columns[xSize:]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
"""
Argumentos de Ejecucion
"""
arg1 = int(sys.argv[1]) # 0 al 3 - proceso
arg2 = int(sys.argv[2]) # 0 al 3 - y_predict
arg3 = int(sys.argv[3]) # 0 al 17 (classifier) 0 al 13 (regressor)
arg4 = 1 # 1:classifier, 0: regression
listProcess = ["randomized", "exhaustive", "edas", "eas"]
listPredict = ["FLOOR", "BUILDINGID", "LATITUDE", "LONGITUDE"]
process = listProcess[arg1]
y_column = listPredict[arg2]
idModelo = arg3

if(arg4):
    estimadorDictionary = getClassifierModels(includeEnsambled=True)
    modelNameList = getClassifierNames(includeEnsambled=True)
else:
    estimadorDictionary = getRegressorModels(includeEnsambled=True)
    modelNameList = getRegressorNames(includeEnsambled=True)

modelName = modelNameList[idModelo]
hypSwitcher = HyperparameterSwitcher(modelName)
estimador = estimadorDictionary[modelName]
parametros = hypSwitcher.getHyperparameters()()
searchParams = hypSwitcher.getHeurisctics()()

"""
Hyperparameter Search
"""
ev = Evaluator(X_train, y_train[y_column])
ev.setEstimador(estimador)
ev.setParams(parametros)
ev.setTypeSearch(process)
ev.fit(scoring='accuracy', n_jobs=cpu_count(), kargs=searchParams)
# Guardar Modelo en formato csv
ev.saveDataFrame(modelName + y_column)