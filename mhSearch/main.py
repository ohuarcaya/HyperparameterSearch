import sys
import time
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
xSize = 1055
df = pd.read_csv("data/filtred.csv")
X = df[df.columns[:xSize]]
Y = df[df.columns[xSize:]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
"""
Argumentos de Ejecucion
"""#1 2 9
arg1 = int(sys.argv[1]) # 0:randomized, 1:exhaustive, 2:edas, 3:eas
arg2 = int(sys.argv[2]) # 0:FLOOR, 1:BUILDINGID, 2:LATITUDE, 3:LONGITUDE
arg3 = int(sys.argv[3]) # 0 al 17 (classifier) 0 al 13 (regressor)
arg4 = 0 # 1:classifier, 0: regression
# arg4 = True if arg2<=1 else False
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
ev = Evaluator(X_train, y_train[y_column], seed)
ev.setEstimador(estimador)
ev.setParams(parametros)
ev.setTypeSearch(process)
n_jobs = cpu_count() # 1
start_time = time.time()
ev.fit(scoring='mse', n_jobs=n_jobs, kargs=searchParams)
print("--- %s seconds ---" % (time.time() - start_time))
# Guardar Modelo en formato csv
ev.saveDataFrame(modelName + y_column)