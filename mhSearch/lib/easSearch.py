"""
python -m pip uninstall deap
python -m pip install --upgrade --upgrade-strategy "eager" --force-reinstall
     --ignore-installed --compile --process-dependency-links --no-binary :all: deap
pip install --upgrade --upgrade-strategy "eager" --force-reinstall
     --ignore-installed --compile --process-dependency-links --no-binary :all: deap
"""
import os
import time
import warnings
import numpy as np
import random as rnd
import pandas as pd
from collections import defaultdict
# Librería Genética
from deap import base
from deap import creator
from deap import tools # Import warning
from deap import algorithms
# Subfunciones de estimadores
from sklearn.base import clone # Import warning
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py][30]
from sklearn.base import is_classifier
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py][535]
from sklearn.model_selection._validation import _fit_and_score # Import warning
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py][346]
from sklearn.model_selection._search import BaseSearchCV
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py][386]
from sklearn.model_selection._search import check_cv
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_split.py][1866]
from sklearn.model_selection._search import _check_param_grid
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py][343]
from sklearn.metrics.scorer import check_scoring
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py][250]
from sklearn.utils.validation import _num_samples
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py][105]
from sklearn.utils.validation import indexable
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py][208]
from multiprocessing import Pool, Manager, cpu_count
# Selección para estimadores
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Metricas para estimadores
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
# Estimadores
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def _get_param_types_maxint(params):
    params_data = list(params.items())  # name_values
    params_type = [isinstance(params[key][0], float) + 1 for key in params.keys()]  # gene_type
    params_size = [len(params[key]) - 1 for key in params.keys()]  # maxints
    return params_data, params_type, params_size


def _initIndividual(pcls, maxints):
    part = pcls(rnd.randint(0, maxint) for maxint in maxints)
    return part


def _mutIndividual(individual, maxints, prob_mutacion):
    for i in range(len(maxints)):
        if rnd.random() < prob_mutacion:
            individual[i] = rnd.randint(0, maxints[i])
    return individual,


def _cxIndividual(ind1, ind2, prob_cruce, gene_type):
    CATEGORICO = 1  # int o str
    NUMERICO = 2  # float
    for i in range(len(ind1)):
        if rnd.random() < prob_cruce:
            if gene_type[i] == CATEGORICO:
                ind1[i], ind2[i] = ind2[i], ind1[i]
            else:
                sorted_ind = sorted([ind1[i], ind2[i]])
                ind1[i] = rnd.randint(sorted_ind[0], sorted_ind[1])
                ind2[i] = rnd.randint(sorted_ind[0], sorted_ind[1])
    return ind1, ind2


def _individual_to_params(individual, name_values):
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

def _evalFunction(individual, name_values, X, y, scorer, cv, uniform, fit_params,
                  verbose=0, error_score='raise', score_cache={}, result_cache=[]):
    parameters = _individual_to_params(individual, name_values)
    nombreModelo = str(individual.est).split('(')[0] # individual.est.__class__.__name__
    score = 0
    paramkey = nombreModelo+str(individual)
    if 'genCount' in score_cache:
        score_cache['genCount'] = score_cache['genCount'] + 1
    else:
        score_cache['genCount'] = 1
    if paramkey in score_cache:
        score = score_cache[paramkey]
    else:
        try:
            resultIndividuo = []
            scorer = scoring_reg = { 'mae': make_scorer(mae), 'mse': make_scorer(mse), 'approach': make_scorer(distance2d) }
            for train, test in cv.split(X, y):
                resultIndividuo.append(_fit_and_score(estimator=individual.est, X=X, y=y, scorer=scorer,  parameters=parameters,
                        train=train, test=test, verbose=verbose, fit_params=None, return_times=True))
            df = pd.DataFrame(list(map(lambda x: _evalfs(x), resultIndividuo)))
            accuracy = np.array(resultIndividuo)[:, 0]  # accuracy
            runtime = np.array(resultIndividuo)[:, 2] + np.array(resultIndividuo)[:, 1]  # runtime train+test
            score = df['approach'].mean()
            score_cache[paramkey] = score
            dict_result = parameters
            dict_result['Approach'] = score
            dict_result['stdApproach'] = df['approach'].std()
            dict_result['MSE'] = df['mse'].mean()
            dict_result['stdMSE'] = df['mse'].std()
            dict_result['MAE'] = df['mae'].mean()
            dict_result['stdMAE'] = df['mae'].std()
            dict_result['Runtime'] = df['time'].mean()
            dict_result['stdRuntime'] = df['time'].std()
            dict_result['genCount'] = score_cache['genCount']
            result_cache.append(dict_result)
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
            dict_result['genCount'] = score_cache['genCount']
            result_cache.append(dict_result)
    return (score,)

def _evalFunctionClassifier(individual, name_values, X, y, scorer, cv, uniform, fit_params,
                  verbose=0, error_score='raise', score_cache={}, result_cache=[]):
    parameters = _individual_to_params(individual, name_values)
    nombreModelo = str(individual.est).split('(')[0]
    score = 0
    paramkey = nombreModelo+str(individual)
    if 'genCount' in score_cache:
        score_cache['genCount'] = score_cache['genCount'] + 1
    else:
        score_cache['genCount'] = 1
    if paramkey in score_cache:
        score = score_cache[paramkey]
    else:
        try:
            resultIndividuo = []
            scorer = check_scoring(individual.est, scoring="accuracy")
            for train, test in cv.split(X, y):
                resultIndividuo.append(_fit_and_score(estimator=individual.est, X=X, y=y, scorer=scorer,
                            train=train, test=test, verbose=verbose, parameters=parameters, fit_params=None, return_times=True))
            accuracy = np.array(resultIndividuo)[:, 0]  # accuracy
            runtime = np.array(resultIndividuo)[:, 2] + np.array(resultIndividuo)[:, 1]  # runtime train+test
            score = accuracy.mean()
            score_cache[paramkey] = score
            dict_result = parameters
            dict_result['Accuracy'] = score
            dict_result['stdAccuracy'] = accuracy.std()
            dict_result['Runtime'] = runtime.mean()
            dict_result['stdRuntime'] = runtime.std()
            dict_result['genCount'] = score_cache['genCount']
            result_cache.append(dict_result)
        except Exception as ex:
            print(ex)
            score_cache[paramkey] = 0
            dict_result = parameters
            dict_result['Accuracy'] = 0
            dict_result['stdAccuracy'] = 0
            dict_result['Runtime'] = 0
            dict_result['stdRuntime'] = 0
            dict_result['genCount'] = score_cache['genCount']
            result_cache.append(dict_result)
    return (score,)


class GeneticSearchCV:
    def __init__(self, estimator, params, scoring=None, cv=4,
            refit=True, verbose=False, population_size=20,
            gene_mutation_prob=0.1, gene_crossover_prob=0.5,
            tournament_size=3, generations_number=10, gene_type=None,
            n_jobs=1, uniform=True, error_score='raise',
            fit_params={}):
        # Parámetros iniciales
        self.estimator = estimator
        self.params = params
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.verbose = verbose
        self.population_size = population_size
        self.gene_mutation_prob = gene_mutation_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.tournament_size = tournament_size
        self.generations_number = generations_number
        self.gene_type = gene_type
        self.n_jobs = n_jobs
        self.uniform = uniform
        self.error_score = error_score
        self.fit_params = fit_params
        # Parámetros adicionales
        self._individual_evals = {}
        self.all_history_ = None
        self.all_logbooks_ = None
        self._cv_results = None
        self.best_score_ = None
        self.best_params_ = None
        self.scorer_ = None
        self.__manager = Manager()
        self.score_cache = self.__manager.dict()
        self.result_cache = self.__manager.list()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, est=clone(self.estimator), fitness=creator.FitnessMax)
    @property
    def cv_results_(self):
        if self._cv_results is None:
            out = defaultdict(list)
            gen = self.all_history_
            idxs, individuals, each_scores = zip(*[(idx, indiv, np.mean(indiv.fitness.values))
											for idx, indiv in list(gen.genealogy_history.items())
											if indiv.fitness.valid and not 
                                            np.all(np.isnan(indiv.fitness.values))])
            name_values, _, _ = _get_param_types_maxint(self.params)
            out['index'] += idxs
            out['params'] += [_individual_to_params(indiv, name_values) for indiv in individuals]
            out['mean_test_score'] += [np.nanmean(scores) for scores in each_scores]
            out['std_test_score'] += [np.nanstd(scores) for scores in each_scores]
            out['min_test_score'] += [np.nanmin(scores) for scores in each_scores]
            out['max_test_score'] += [np.nanmax(scores) for scores in each_scores]
            out['nan_test_score?'] += [np.any(np.isnan(scores)) for scores in each_scores]
            self._cv_results = out
        return self._cv_results
    @property
    def best_index_(self):
        return np.argmax(self.cv_results_['max_test_score'])
    def fit(self, X, y):
        self.best_estimator_ = None
        self.best_mem_score_ = float("-inf")
        self.best_mem_params_ = None
        _check_param_grid(self.params)
        self._fit(X, y, self.params)
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_mem_params_)
            if self.fit_params is not None:
                self.best_estimator_.fit(X, y, **self.fit_params)
            else:
                self.best_estimator_.fit(X, y)
    def _fit(self, X, y, parameter_dict):
        self._cv_results = None
        # self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        n_samples = _num_samples(X)
        if _num_samples(y) != n_samples:
            raise ValueError('Target [y], data [X] no coinciden')
        #self.cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))
        # self.cv = KFold(n_splits=10, shuffle=True, random_state=self.seed) # Just for
        toolbox = base.Toolbox()
        name_values, self.gene_type, maxints = _get_param_types_maxint(parameter_dict)
        if self.verbose:
            print("Tipos: %s, rangos: %s" % (self.gene_type, maxints))
        # registro de función Individuo
        toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
        # registro de función Población
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # Paralelísmo, create pool
        if not isinstance(self.n_jobs, int):
            self.n_jobs=1
        pool = Pool(self.n_jobs)
        toolbox.register("map", pool.map)
        # registro de función Evaluación
        toolbox.register("evaluate", _evalFunction,
						name_values=name_values, X=X, y=y,
						scorer=self.scorer_, cv=self.cv, uniform=self.uniform, verbose=self.verbose,
						error_score=self.error_score, fit_params=self.fit_params,
						score_cache=self.score_cache, result_cache=self.result_cache)
        # registro de función Cruce
        toolbox.register("mate", _cxIndividual, prob_cruce=self.gene_crossover_prob, 
                        gene_type=self.gene_type)
        # registro de función Mutación
        toolbox.register("mutate", _mutIndividual, prob_mutacion=self.gene_mutation_prob, maxints=maxints)
        # registro de función Selección
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        # Creación de Población
        pop = toolbox.population(n=self.population_size)
        # Mejor Individuo que ha existido
        hof = tools.HallOfFame(1)
        # Stats
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.nanmean)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        stats.register("std", np.nanstd)
        # Genealogía
        hist = tools.History()
        # Decoración de operadores de variaznza
        toolbox.decorate("mate", hist.decorator)
        toolbox.decorate("mutate", hist.decorator)
        hist.update(pop)
        # Posibles combinaciones
        if self.verbose:
            print('--- Evolve in {0} possible combinations ---'.format(np.prod(np.array(maxints) + 1)))
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=self.gene_crossover_prob, 
                                        mutpb=self.gene_mutation_prob,
										ngen=self.generations_number, stats=stats,
										halloffame=hof, verbose=self.verbose)
        # Save History
        self.all_history_ = hist
        self.all_logbooks_ = logbook
        # Mejor score y parametros
        current_best_score_ = hof[0].fitness.values[0]
        current_best_params_ = _individual_to_params(hof[0], name_values)
        if self.verbose:
            print("Best individual is: %s\nwith fitness: %s" % (
                current_best_params_, current_best_score_))
        if current_best_score_ > self.best_mem_score_:
            self.best_mem_score_ = current_best_score_
            self.best_mem_params_ = current_best_params_
        # fin paralelización, close pool
        pool.close()
        pool.join()
        self.best_score_ = current_best_score_
        self.best_params_ = current_best_params_
