import sys
sys.path.append("./lib/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
process = 'exhaustive'
for idModeloPrueba in range(len(estimadorDictionary)):
    try:
        result = {}
        modelName = getClassifierNames(includeEnsambled=True)[idModeloPrueba]
        estimador = estimadorDictionary[modelName]
        parametros = hypSwitcher.getHyperparameters(modelName)()
        grid_search = GridSearchCV(estimador, param_grid=parametros, 
                                        cv=kf, scoring="accuracy", 
                                        return_train_score=False, n_jobs=-1)
        grid_search.fit(X_train, y_train.FLOOR)
        result[modelName] = grid_search.cv_results_
        df1 = pd.DataFrame(np.array([result[modelName]['mean_test_score'], result[modelName]['std_test_score'],
                                    result[modelName]['mean_fit_time'], result[modelName]['std_fit_time'],
                                    result[modelName]['mean_score_time'], result[modelName]['std_score_time']
                                    ]).T, columns = ['Accuracy', 'stdAccuracy', 'FitTime', 'stdFitTime', 'ScoreTime', 'stdScoreTime'])
        df2 = pd.DataFrame(result[modelName]['params'])
        dff = pd.concat([df1,df2], axis=1).sort_values(['Accuracy', 'FitTime'], ascending=[False, True])
        dff.to_csv("result/" + process + "/" + modelName + "floor.csv", index=False)
    except:
        dff = pd.DataFrame()
        dff.to_csv("result/" + process + "/" + modelName + "-floor-FALLO.csv", index=False)

for idModeloPrueba in range(len(estimadorDictionary)):
    try:
        result = {}
        modelName = getClassifierNames(includeEnsambled=True)[idModeloPrueba]
        estimador = estimadorDictionary[modelName]
        parametros = hypSwitcher.getHyperparameters(modelName)()
        grid_search = GridSearchCV(estimador, param_grid=parametros, 
                                        cv=kf, scoring="accuracy", 
                                        return_train_score=False, n_jobs=-1)
        grid_search.fit(X_train, y_train.BUILDINGID)
        result[modelName] = grid_search.cv_results_
        df1 = pd.DataFrame(np.array([result[modelName]['mean_test_score'], result[modelName]['std_test_score'],
                                    result[modelName]['mean_fit_time'], result[modelName]['std_fit_time'],
                                    result[modelName]['mean_score_time'], result[modelName]['std_score_time']
                                    ]).T, columns = ['Accuracy', 'stdAccuracy', 'FitTime', 'stdFitTime', 'ScoreTime', 'stdScoreTime'])
        df2 = pd.DataFrame(result[modelName]['params'])
        dff = pd.concat([df1,df2], axis=1).sort_values(['Accuracy', 'FitTime'], ascending=[False, True])
        dff.to_csv("result/" + process + "/" + modelName + "building.csv", index=False)
    except:
        dff = pd.DataFrame()
        dff.to_csv("result/" + process + "/" + modelName + "-building-FALLO.csv", index=False)