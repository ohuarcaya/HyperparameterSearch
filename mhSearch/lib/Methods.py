import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import train_test_split


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
            # error = distance_error(estimator, X, y)
            score = accuracy.mean()
            score_cache[paramkey] = score
            #dict_result = {}
            #dict_result['Modelo'] = nombreModelo
            #dict_result['Parametros'] = params
            #dict_result['Accuracy'] = score
            #dict_result['stdAccuracy'] = accuracy.std()
            #dict_result['Runtime'] = runtime.mean()
            #dict_result['accuracy_values'] = accuracy
            #dict_result['runtime_values'] = runtime
            dict_result = params
            dict_result['Accuracy'] = score
            dict_result['stdAccuracy'] = accuracy.std()
            dict_result['Runtime'] = runtime.mean()
            dict_result['stdRuntime'] = runtime.std()
            dict_result['generacion'] = generacion
            resultados.append(dict_result)
        return score


def _individual_to_params(individual, parametros):
    individual = np.int32(individual)
    name_values = list(parametros.items())
    return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))


def prettyPrint(indice, individual, parametros):
    dict_result = _individual_to_params(individual, parametros)
    dict_result['Accuracy'] = individual[-1]
    result = pd.DataFrame([dict_result]).to_string(index=False).split('\n')
    if(indice == 0):
        print('indice\t' + result[0])
    print(str(indice) + '\t' + result[1])
