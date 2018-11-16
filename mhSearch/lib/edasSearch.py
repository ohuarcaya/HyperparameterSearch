import numpy as np
import pandas as pd
import itertools as it
from Methods import prettyPrint
from multiprocessing import Pool, Manager, cpu_count
"""
Class EdasHyperparameterSearch
    of: "funcion objetivo"
    parametros: "campo de hiperparametros, type:dict"
    estimator: "modelo de prediccion para los cuales se insertaron los hiperparametros"
    iterations: "numero de generaciones"
    sample_size: "tamanho de la poblacion"
    select_ratio: "elitismo, porcentaje de supervivencia"
    debug: "mostrar avance"
"""


class EdasHyperparameterSearch:
    def __init__(self, of, parametros, estimator, iterations=10, sample_size=50,
                 select_ratio=0.3, debug=False, n_jobs=1, type="accuracy"):
        # Algorithm parameters
        self.iterations = 1 if iterations <= 1 else iterations - 1
        self.sample_size = sample_size
        self.select_ratio = select_ratio
        self.epsilon = 10e-6
        # class members
        self.class_method = of
        self.objective_function = of.getModelAccuracy if type=="accuracy" else of.getModelMSE
        self.sample = []
        self.means = []
        self.stdevs = []
        self.debug = debug
        # aditional parameters
        self.parametros = parametros
        self.estimator = estimator
        self.__manager = Manager()
        self.score_cache = self.__manager.dict()
        self.resultados = self.__manager.list()
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.dimensions = len(parametros)

    def sample_sort(self):
        self.sample = self.sample[np.argsort(self.sample[:, -1], 0)]

    def dispersion_reduction(self):
        self.sample_sort()
        nb = int(np.floor(self.sample_size * self.select_ratio))
        self.sample = self.sample[self.sample_size - nb:]
        if self.debug:
            print("dispersion reduction")
            print(str(self.sample))
            print

    def estimate_parameters(self):
        mat = self.sample  # self.sample[:, :self.dimensions]
        self.means = np.mean(mat, 0)
        self.stdevs = np.std(mat, 0)
        if self.debug:
            print("estimate parameters")
            print("\tmean=" + str(self.means))
            print("\tstd-dev=" + str(self.stdevs))
            print

    def draw_sample(self):
        # for each variable to optimize
        self.stdevs = ((self.stdevs == 0) * self.epsilon) + self.stdevs
        self.sample = np.floor(np.random.normal(
            self.means, self.stdevs, size=(self.sample_size, self.dimensions + 1)))
        var = (np.max(self.sample, 0) - np.min(self.sample, 0))
        var = var + (var == 0) * self.epsilon
        self.sample = np.floor(
            ((self.sample - np.min(self.sample, 0)) / var) * (self.tope_params - 1))
        self.sample = np.unique(self.sample, axis=0)
        if self.debug:
            print("draw sample")
            print(self.sample)
            print

    def evaluate(self):
        _pool = Pool(self.n_jobs)
        # iterador de parametros de función objetivo multiproceso
        _iterable = it.product([self.parametros], np.int32(self.sample[:, :self.dimensions]),
                               [self.score_cache], [self.resultados], [self.generacion])  # [self.estimator],
        self.sample[:, -1] = _pool.starmap(self.objective_function, _iterable)
        _pool.close()
        _pool.join()
        if self.debug:
            print("evaluate")
            print(self.sample)
            print

    def run(self):
        self.sample = np.random.rand(self.sample_size, self.dimensions + 1)  # uniform initialization
        # cosmetic
        self.params_size = [len(self.parametros[key]) -
                            1 for key in self.parametros.keys()]  # maxints
        self.tope_params = np.array(self.params_size + [-1]) + 1
        self.sample = np.floor(self.sample * self.tope_params)
        self.sample = np.unique(self.sample, axis=0)
        if self.debug:
            print("initialization")
            print(self.sample)
            print
        i = 0
        self.generacion = 0
        self.evaluate()  # Multi process
        self.sample_sort()
        prettyPrint(i, self.sample[-1], self.parametros)
        while i < self.iterations:
            if self.debug:
                print("iteration", i)
                print
            i += 1
            self.dispersion_reduction()
            self.estimate_parameters()
            self.draw_sample()
            self.generacion = i
            self.evaluate()
            self.sample_sort()
            prettyPrint(i, self.sample[-1], self.parametros)
