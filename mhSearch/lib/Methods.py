import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import train_test_split
from sklearn import metrics as scoreMetrics


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
                dict_result['Approach'] = score
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
                dict_result['Approach'] = 0
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