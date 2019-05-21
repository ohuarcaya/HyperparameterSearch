import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import train_test_split
from sklearn import metrics as scoreMetrics
from sklearn.metrics import make_scorer

class GeneralMethods:
    def __init__(self, estimador, X, y, test_size=0.2, seed=7):
        self.seed = seed
        self.setEstimador(estimador)
        #self._createDataset(urlDataset=urlDataset, test_size=test_size)
        self.X  = X
        self.y = y

    def setEstimador(self, estimador):
        self.estimator = estimador

    def _createDataset(self, urlDataset, test_size=0.2):
        dataset = pd.read_csv(urlDataset)
        names_ = dataset.columns.values
        dataset = utils.shuffle(
            dataset, random_state=self.seed).reset_index(drop=True)
        dataset = dataset.apply(pd.to_numeric)
        X = dataset[names_[:-1]]
        y = dataset[names_[-1]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed)

    def distance_error(self):
        self.estimator.fit(self.X_train, self.y_train)
        distanciaEntrePuntos = 1.5
        y_pred = self.estimator.predict(self.X_test)
        x1 = np.int32((y_pred + 2) % 3)
        y1 = np.int32((y_pred - 1) / 3)
        x2 = np.int32((self.y_test + 2) % 3)
        y2 = np.int32((self.y_test - 1) / 3)
        # pasar variacion a distancias metros
        vx = np.abs(x1 - x2) * distanciaEntrePuntos
        vy = np.abs(y1 - y2) * distanciaEntrePuntos
        err_distance = np.mean(np.sqrt(vx*vx + vy*vy))
        return err_distance

    def getModelAccuracy(self, parametros, individual, score_cache, resultados, generacion):
        params = _individual_to_params(individual, parametros)
        score = 0
        scoring = "accuracy"
        nombreModelo = str(self.estimator).split('(')[0]
        paramkey = nombreModelo + str(np.int32(individual))
        if paramkey in score_cache:
            score = score_cache[paramkey]
        else:
            try:
                resultIndividuo = []
                cv = KFold(n_splits=10, shuffle=False)
                scorer = check_scoring(self.estimator, scoring=scoring)
                for train, test in cv.split(self.X, self.y):
                    resultIndividuo.append(_fit_and_score(estimator=self.estimator, 
                                    X=self.X, y=self.y, scorer=scorer, parameters=params,
                                    train=train, test=test, verbose=0, fit_params=None, return_times=True))
                accuracy = np.array(resultIndividuo)[:, 0]  # accuracy
                runtime = np.array(resultIndividuo)[
                    :, 2] + np.array(resultIndividuo)[:, 1]  # runtime train+test
                score = accuracy.mean()
                score_cache[paramkey] = score
                dict_result = params
                dict_result['Accuracy'] = score
                dict_result['stdAccuracy'] = accuracy.std()
                dict_result['Runtime'] = runtime.mean()
                dict_result['stdRuntime'] = runtime.std()
                dict_result['generacion'] = generacion
                resultados.append(dict_result)
            except Exception as ex:
                print(ex)
                score_cache[paramkey] = 0
                dict_result = params
                dict_result['Accuracy'] = 0
                dict_result['stdAccuracy'] = 0
                dict_result['Runtime'] = 0
                dict_result['stdRuntime'] = 0
                dict_result['generacion'] = generacion
                resultados.append(dict_result)
        return score

    def getModelApproach(self, parametros, individual, score_cache, resultados, generacion):
        params = _individual_to_params(individual, parametros)
        score = 0
        scoring = "mse"
        nombreModelo = str(self.estimator).split('(')[0]
        paramkey = nombreModelo + str(np.int32(individual))
        if paramkey in score_cache:
            score = score_cache[paramkey]
        else:
            try:
                resultIndividuo = []
                cv = KFold(n_splits=10, shuffle=True, random_state=self.seed)
                scorer = scoring_reg = { 'mae': make_scorer(mae), 'mse': make_scorer(mse), 'approach': make_scorer(distance2d) }
                for train, test in cv.split(self.X, self.y):
                    resultIndividuo.append(_fit_and_score(estimator=self.estimator, X=self.X, y=self.y, scorer=scorer, parameters=params,
                                    train=train, test=test, verbose=0, fit_params=None, return_times=True))
                df = pd.DataFrame(list(map(lambda x: _evalfs(x), resultIndividuo)))
                score = df['approach'].mean()
                score_cache[paramkey] = score
                dict_result = params
                dict_result['Accuracy'] = score
                dict_result['stdApproach'] = df['approach'].std()
                dict_result['MSE'] = df['mse'].mean()
                dict_result['stdMSE'] = df['mse'].std()
                dict_result['MAE'] = df['mae'].mean()
                dict_result['stdMAE'] = df['mae'].std()
                dict_result['Runtime'] = df['time'].mean()
                dict_result['stdRuntime'] = df['time'].std()
                dict_result['generacion'] = generacion
                resultados.append(dict_result)
            except Exception as ex:
                print(ex)
                score_cache[paramkey] = 0
                dict_result = params
                dict_result['Accuracy'] = 0
                dict_result['stdApproach'] = 0
                dict_result['Runtime'] = 0
                dict_result['stdRuntime'] = 0
                dict_result['MSE'] = 100
                dict_result['stdMSE'] = 100
                dict_result['MAE'] = 100
                dict_result['stdMAE'] = 100
                dict_result['generacion'] = generacion
                resultados.append(dict_result)
        return score


def _individual_to_params(individual, parametros):
    individual = np.int32(individual)
    name_values = list(parametros.items())
    return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))
    # try:
    #     response  = dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))
    # except e:
    #     print(individual)
    #     print(name_values)
    #     print(e)
    #     response  = dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))
    # return response
    individual = np.int32(individual)
    name_values = list(parametros.items())
    try:
        return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))
    except Exception as ex:
        print(individual)
        print(name_values)
        return {}


def distance2d(y_true, y_pred):
    y_true = np.array(list(y_true[0]) if (list(y_true)==[0]) else list(y_true))
    #y_pred = np.array(list(y_pred[0]) if (list(y_pred)==[0]) else list(y_pred))
    #_range = np.concatenate((y_true,y_pred))
    _range = [4864745, 4865018] if np.min(y_true)>0 else [-7696, -7300]
    _limit = (np.max(_range)/2 - np.min(_range)/2)**2
    out = (_limit - mse(y_true,y_pred))/_limit
    if (out<0):
        return 0
    else:
        return out

def mse(y_true, y_pred):
    return scoreMetrics.mean_squared_error(y_true, y_pred)
    
def mae(y_true, y_pred):
    return scoreMetrics.mean_absolute_error(y_true, y_pred)

def _evalfs(x):
    d = x[0]
    d['time'] = x[1] + x[2]
    return d

def prettyPrint(indice, individual, parametros):
    dict_result = _individual_to_params(individual, parametros)
    dict_result['Accuracy'] = individual[-1]
    result = pd.DataFrame([dict_result]).to_string(index=False).split('\n')
    if(indice == 0):
        print('indice\t' + result[0])
    print(str(indice) + '\t' + result[1])

"""
scorer = check_scoring(modelo_lr, scoring='neg_mean_squared_error')

from sklearn.datasets import load_boston
boston = load_boston()
scorer = scoring_reg
_X = pd.DataFrame(boston.data)
_y = pd.DataFrame(boston.target)
cv = KFold(n_splits=10, shuffle=True, random_state=7)

resultIndividuo = []
for train, test in cv.split(_X, _y):
    resultIndividuo.append(_fit_and_score(estimator=modelo_lr, X=_X, y=_y, scorer=scorer, parameters=params,
                    train=train, test=test, verbose=0, fit_params=None, return_times=True))

def _f(x):
    d = x[0]
    d['time'] = x[1] + x[2]
    return d
pd.DataFrame(list(map(lambda x: _evalfs(x), resultIndividuo)))
pd.DataFrame(list(map(lambda x: x[0], resultIndividuo)))


- 13 HTC Wildfire S 2.3.5 0,11          [4885]

- 7 GT-S6500 2.3.6 14                   [1596]
- 6 GT-S5360 2.3.6 7                    [1383]

- 23 Transformer TF101 4.0.3 2          [1091]ASUS

- 9 Galaxy Nexus 4.3 0                  [77]
- 8 Galaxy Nexus 4.2.2 10               [913]

- 17 M1005D 4.0.4 13                    [841]HTC
- 22 Orange Monte Carlo 2.3.5 17        [724]*

- 11 HTC One 4.1.2 15                   [498]
- 10 HTC Desire HD 2.3.5 18             [440]
- 12 HTC One 4.2.2 0                    [70]

- 19 Nexus 4 4.2.2 6                    [980]
- 20 Nexus 4 4.3 0                      [213]
- 21 Nexus S 4.1.2 0                    [60]


- 14 LT22i 4.0.4 0,1,9,16               [4863]
- 16 LT26i 4.0.4 3                      [192]
- 15 LT22i 4.1.2 0                      [36]

- 3 GT-I9100 4.0.4 5                    [610]
- 1 GT-I8160 2.3.6 8                    [507]
- 4 GT-I9300 4.1.2 0                    [69]
- 2 GT-I8160 4.1.2 0                    [52]
- 5 GT-I9505 4.2.2 0                    [17]

- 24 bq Curie 4.1.1 12                  [437]*
- 18 MT11i 2.3.4 4                      [374]SONY
- 0 Celkon A27 4.0.4(6577) 0            [120]

"""

"""
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
xSize = 1055
df = pd.read_csv("data/filtred.csv")
X = df[df.columns[:xSize]]
Y = df[df.columns[xSize:]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
arg1 = 2 # int(sys.argv[1]) # 0:randomized, 1:exhaustive, 2:edas, 3:eas
arg2 = 2 # int(sys.argv[2]) # 0:FLOOR, 1:BUILDINGID, 2:LATITUDE, 3:LONGITUDE
arg3 = 0 # int(sys.argv[3]) # 0 al 17 (classifier) 0 al 13 (regressor)
arg4 = 0 # 1:classifier, 0: regression
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
ev = Evaluator(X_train, y_train[y_column], seed)
ev.setEstimador(estimador)
ev.setParams(parametros)
ev.setTypeSearch(process)
n_jobs = 1 # cpu_count()
ev.fit(scoring='mse', n_jobs=n_jobs, kargs=searchParams)
# Guardar Modelo en formato csv
ev.saveDataFrame(modelName + y_column)




import numpy as np
import pandas as pd
from Methods import GeneralMethods
from easSearch import GeneticSearchCV
from sklearn.model_selection import KFold
from edasSearch import EdasHyperparameterSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics as scoreMetrics
from sklearn.metrics import make_scorer


def distance2d(y_true, y_pred):
    y_true = np.array(list(y_true[0]) if (list(y_true)==[0]) else list(y_true))
    y_pred = np.array(list(y_pred[0]) if (list(y_pred)==[0]) else list(y_pred))
    _range = np.concatenate((y_true,y_pred))
    _limit = (np.max(_range) - np.min(_range))**2
    return (_limit - mse(y_true,y_pred))/_limit


def mse(y_true, y_pred):
    return scoreMetrics.mean_squared_error(y_true, y_pred)
    

def mae(y_true, y_pred):
    return scoreMetrics.mean_absolute_error(y_true, y_pred)


X = X_train
y = y_train[y_column]
kf = KFold(n_splits=10, shuffle=True, random_state=seed) # Just for
estimador = estimadorDictionary[modelName]
params = parametros    
type = process

scoring = "neg_mean_squared_error"
scoring = { 'mae': make_scorer(mae), 'mse': make_scorer(mse), 'approach': make_scorer(distance2d) }
escv = GridSearchCV(estimador, param_grid=params, cv=kf, scoring=scoring, refit=False
                        return_train_score=False, n_jobs=n_jobs)
escv.fit(X, y)
df1 = pd.DataFrame(np.array([escv.cv_results_['mean_test_score'], escv.cv_results_['std_test_score'],
                            escv.cv_results_['mean_fit_time'], escv.cv_results_['std_fit_time'],
                            escv.cv_results_['mean_score_time'], escv.cv_results_['std_score_time']
                            ]).T, columns = ['Accuracy', 'stdAccuracy', 'FitTime', 'stdFitTime', 
                            'ScoreTime', 'stdScoreTime'])
df2 = pd.DataFrame(escv.cv_results_['params'])
dff = pd.concat([df1,df2], axis=1).sort_values(['Accuracy', 'FitTime'], ascending=[False, True])

"""
"""

dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
 'param_fit_intercept', 'param_normalize', 'params', 'split0_test_mae', 'split1_test_mae',
 'split2_test_mae', 'split3_test_mae', 'split4_test_mae', 'split5_test_mae', 'split6_test_mae',
 'split7_test_mae', 'split8_test_mae', 'split9_test_mae', 'mean_test_mae', 'std_test_mae', 
 'rank_test_mae', 'split0_test_mse', 'split1_test_mse', 'split2_test_mse', 'split3_test_mse',
 'split4_test_mse', 'split5_test_mse', 'split6_test_mse', 'split7_test_mse', 'split8_test_mse', 
 'split9_test_mse', 'mean_test_mse', 'std_test_mse', 'rank_test_mse', 'split0_test_approach', 
 'split1_test_approach', 'split2_test_approach', 'split3_test_approach', 'split4_test_approach', 
 'split5_test_approach', 'split6_test_approach', 'split7_test_approach', 'split8_test_approach', 
 'split9_test_approach', 'mean_test_approach', 'std_test_approach', 'rank_test_approach']


 df1 = pd.DataFrame(np.array([escv.cv_results_['mean_test_approach'], escv.cv_results_['std_test_approach'],
                            escv.cv_results_['mean_test_mse'], escv.cv_results_['std_test_mse'],
                            escv.cv_results_['mean_test_mae'], escv.cv_results_['std_test_mae'],
                            escv.cv_results_['mean_fit_time'], escv.cv_results_['std_fit_time'],
                            escv.cv_results_['mean_score_time'], escv.cv_results_['std_score_time']
                            ]).T, columns = ['Accuracy', 'stdAccuracy', 'Mse', 'stdMse', 'Mae', 'stdMae', 'FitTime', 'stdFitTime', 
                            'ScoreTime', 'stdScoreTime'])

"""