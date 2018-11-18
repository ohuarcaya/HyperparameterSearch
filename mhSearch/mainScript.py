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

seed = 9
xSize = 1041
df = pd.read_csv("data/filtred.csv")
X = df[df.columns[:xSize]]
Y = df[df.columns[xSize:]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

estimadorDictionary = getClassifierModels(includeEnsambled=True)
hypSwitcher = HyperparameterSwitcher()
modelNameList = getClassifierNames(includeEnsambled=True)
#for idModeloPrueba in range(len(estimadorDictionary)):
idModelo = 7
process = 'edas'
searchParams = {}
searchParams['ngen'] = 2 # rand, eas, edas
searchParams['psize'] = 3 # eas, edas
searchParams['elit'] = 3 # eas
searchParams['pelit'] = 0.5 # edas

modelName = modelNameList[idModelo]
estimador = estimadorDictionary[modelName]
parametros = hypSwitcher.getHyperparameters(modelName)(isDummy = False)
c = Evaluator(X_train, y_train.FLOOR)
c.setEstimador(estimador)
c.setParams(parametros)
c.setTypeSearch(process)
c.fit(scoring='accuracy', n_jobs=cpu_count(), kargs=searchParams)

c.saveDataFrame(modelName + "FLOOR")

