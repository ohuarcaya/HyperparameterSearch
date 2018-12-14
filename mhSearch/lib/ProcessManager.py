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

class Evaluator:
    def __init__(self, X, y, seed):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=10, shuffle=True, random_state=seed) # Just for

    def setEstimador(self, estimador):
        self.estimador = estimador

    def setParams(self, params):
        self.params = params
    
    def setTypeSearch(self, type):
        self.type = type

    def fit(self, scoring ='accuracy', n_jobs = 1, kargs={}):
        self.complete = ''
        generaciones = kargs['ngen'] if 'ngen' in kargs.keys() else 10 # rand, eas, edas
        pop_size = kargs['psize'] if 'psize' in kargs.keys() else 10 # eas, edas
        elitismo = kargs['elit'] if 'elit' in kargs.keys() else 3 # eas
        prob_elitismo = kargs['pelit'] if 'pelit' in kargs.keys() else 0.5 # edas
        if(scoring!="accuracy"):
            scoring = { 'mae': make_scorer(mae), 'mse': make_scorer(mse), 'approach': make_scorer(distance2d) }
        # try:
        if (self.type == 'eas'):
            agcv = GeneticSearchCV(self.estimador, self.params, cv=self.kf, 
                    n_jobs=n_jobs, verbose=1, scoring=scoring, refit=False, tournament_size=elitismo,
                    generations_number=generaciones, population_size=pop_size)
            agcv.fit(self.X, self.y)
            self.dff = pd.DataFrame(list(agcv.result_cache)).sort_values(['Accuracy'], 
                    ascending=False).reset_index(drop=True)
        if (self.type == 'edas'):
            gm = GeneralMethods(self.estimador, self.X, self.y)
            edcv = EdasHyperparameterSearch(gm, self.params, self.estimador, 
                    iterations=generaciones, sample_size=pop_size, select_ratio=prob_elitismo, 
                    debug=False, n_jobs=n_jobs, type=scoring)
            edcv.run()
            self.dff = pd.DataFrame(list(edcv.resultados)).sort_values(['Accuracy'], 
                    ascending=False).reset_index(drop=True)
        if (self.type == 'exhaustive'):
            escv = GridSearchCV(self.estimador, param_grid=self.params, cv=self.kf, scoring=scoring, refit=False,
                                    return_train_score=False, n_jobs=n_jobs)
            escv.fit(self.X, self.y)
            if(scoring=="accuracy"):
                df1 = pd.DataFrame(np.array([escv.cv_results_['mean_test_score'], escv.cv_results_['std_test_score'],
                                            escv.cv_results_['mean_fit_time'], escv.cv_results_['std_fit_time'],
                                            escv.cv_results_['mean_score_time'], escv.cv_results_['std_score_time']
                                            ]).T, columns = ['Accuracy', 'stdAccuracy', 'FitTime', 'stdFitTime', 
                                            'ScoreTime', 'stdScoreTime'])
            else:
                df1 = pd.DataFrame(np.array([escv.cv_results_['mean_test_approach'], escv.cv_results_['std_test_approach'],
                            escv.cv_results_['mean_test_mse'], escv.cv_results_['std_test_mse'],
                            escv.cv_results_['mean_test_mae'], escv.cv_results_['std_test_mae'],
                            escv.cv_results_['mean_fit_time'], escv.cv_results_['std_fit_time'],
                            escv.cv_results_['mean_score_time'], escv.cv_results_['std_score_time']
                            ]).T, columns = ['Accuracy', 'stdAccuracy', 'Mse', 'stdMse', 'Mae', 'stdMae', 'FitTime', 'stdFitTime', 
                            'ScoreTime', 'stdScoreTime'])
            df2 = pd.DataFrame(escv.cv_results_['params'])
            self.dff = pd.concat([df1,df2], axis=1).sort_values(['Accuracy', 'FitTime'], ascending=[False, True])
        if (self.type == 'randomized'):
            rscv = RandomizedSearchCV(self.estimador, param_distributions=self.params, 
                                n_iter=generaciones, cv=self.kf, scoring=scoring, refit=False,
                                return_train_score=False, n_jobs=n_jobs)
            rscv.fit(self.X, self.y)
            if(scoring=="accuracy"):
                df1 = pd.DataFrame(np.array([rscv.cv_results_['mean_test_score'], rscv.cv_results_['std_test_score'],
                                            rscv.cv_results_['mean_fit_time'], rscv.cv_results_['std_fit_time'],
                                            rscv.cv_results_['mean_score_time'], rscv.cv_results_['std_score_time']
                                            ]).T, columns = ['Accuracy', 'stdAccuracy', 'FitTime', 'stdFitTime', 
                                            'ScoreTime', 'stdScoreTime'])
            else:
                df1 = pd.DataFrame(np.array([rscv.cv_results_['mean_test_approach'], rscv.cv_results_['std_test_approach'],
                            rscv.cv_results_['mean_test_mse'], rscv.cv_results_['std_test_mse'],
                            rscv.cv_results_['mean_test_mae'], rscv.cv_results_['std_test_mae'],
                            rscv.cv_results_['mean_fit_time'], rscv.cv_results_['std_fit_time'],
                            rscv.cv_results_['mean_score_time'], rscv.cv_results_['std_score_time']
                            ]).T, columns = ['Accuracy', 'stdAccuracy', 'Mse', 'stdMse', 'Mae', 'stdMae', 'FitTime', 'stdFitTime', 
                            'ScoreTime', 'stdScoreTime'])
            df2 = pd.DataFrame(rscv.cv_results_['params'])
            self.dff = pd.concat([df1,df2], axis=1).sort_values(['Accuracy', 'FitTime'], ascending=[False, True])
        # except Exception as e:
        #     self.complete = '-fail'
        #     self.dff = pd.DataFrame([{'error': e}])

    def saveDataFrame(self, fileName):
        self.dff.to_csv("result/" + self.type + "/" + fileName + self.complete + ".csv", index=False)
        return self.dff


def distance2d(y_true, y_pred):
    # y_true = np.array(list(y_true[0]) if (list(y_true)==[0]) else list(y_true))
    # y_pred = np.array(list(y_pred[0]) if (list(y_pred)==[0]) else list(y_pred))
    # _range = np.concatenate((y_true,y_pred))
    # _limit = (np.max(_range) - np.min(_range))**2
    # return (_limit - mse(y_true,y_pred))/_limit
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
