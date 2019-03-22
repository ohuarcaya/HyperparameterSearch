import numpy as np
from functools import reduce
class HyperparameterSwitcher:
    def __init__(self, modelName):
        self.modelName = modelName
    
    def getHyperparameters(self):
        methodName = 'Parametros_' + str(self.modelName)
        return getattr(self, methodName, lambda: "Invalid Model Name")
    
    def getHeurisctics(self):
        methodName = 'Heuristics_' + str(self.modelName)
        return getattr(self, methodName, lambda: "Invalid Model Name")

    # classifiers
    def Parametros_LogisticRegression(self):
        parameters = {}
        parameters['multi_class'] = ['ovr', 'multinomial']
        parameters['solver'] = ['newton-cg', 'lbfgs', 'sag']
        parameters['C'] = [0.8, 0.9, 1.0, 1.1, 1.2]
        parameters['warm_start'] = [True, False]
        return parameters

    def Parametros_SGDClassifier(self):
        parameters = {}
        parameters['loss'] = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
        parameters['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
        parameters['alpha'] = np.logspace(-5, 0, 6)
        parameters['learning_rate'] = ['constant', 'invscaling', 'optimal']
        parameters['class_weight'] = [None, 'balanced']
        parameters['warm_start'] = [False, True]
        return parameters

    def Parametros_PassiveAggressiveClassifier(self):
        parameters = {}
        parameters['C'] = [0.8, 0.9, 1.0, 1.1, 1.2]
        parameters['fit_intercept'] = [True, False]
        parameters['warm_start'] = [True, False]
        parameters['class_weight'] = [None, 'balanced']
        parameters['average'] = [True, False]
        return parameters

    def Parametros_MLPClassifier(self):
        parameters = {}
        parameters['hidden_layer_sizes'] = [10, 20, 50, 100]
        parameters['activation'] = ['identity', 'logistic', 'tanh', 'relu']
        parameters['solver'] = ['lbfgs', 'sgd', 'adam']
        parameters['learning_rate'] = ['constant', 'invscaling', 'adaptive']
        parameters['alpha'] = np.logspace(-5, 3, 5)
        return parameters

    def Parametros_LinearDiscriminantAnalysis(self):
        parameters = {}
        parameters['solver'] = ['svd', 'lsqr', 'eigen']
        parameters['shrinkage'] = [None, 'auto']
        return parameters

    def Parametros_QuadraticDiscriminantAnalysis(self):
        parameters = {}
        parameters['reg_param'] = [0.0, 0.1, 0.3, 0.7]
        return parameters

    def Parametros_KNeighborsClassifier(self):
        parameters = {}
        parameters['n_neighbors'] = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
        parameters['weights'] = ['uniform', 'distance']
        parameters['algorithm'] = ['ball_tree', 'kd_tree', 'brute']
        parameters['p'] = [1, 2, 3, 4, 5]
        parameters['leaf_size'] = [10, 30, 50, 70]
        return parameters

    def Parametros_DecisionTreeClassifier(self):
        parameters = {}
        parameters['max_features'] = ['sqrt', 'log2', None]
        parameters['splitter'] = ['best', 'random']
        parameters['max_depth'] = [2, 3, 10, 50, 100]
        parameters['criterion'] = ['gini', 'entropy']
        parameters['class_weight'] = [None, 'balanced']
        return parameters

    def Parametros_GaussianNB(self):
        parameters = {}
        parameters['priors'] = [None]
        return parameters

    def Parametros_BernoulliNB(self):
        parameters = {}
        parameters['alpha'] = [0, 0.3, 0.5, 1.0]
        parameters['fit_prior'] = [True, False]
        return parameters

    def Parametros_MultinomialNB(self):
        parameters = {}
        parameters['alpha'] = [0, 0.3, 0.5, 1.0]
        parameters['fit_prior'] = [True, False]
        return parameters

    def Parametros_SVC(self):
        parameters = {}
        parameters['C'] = [1, 2, 5, 10]
        parameters['decision_function_shape'] = ['ovr']
        parameters['max_iter'] = [10000]
        parameters['shrinking'] = [True, False]
        parameters['kernel'] = ['poly', 'linear', 'rbf', 'sigmoid']
        parameters['degree'] = [1, 2, 3, 4, 5] # conditional for kernel poly
        return parameters

    # ensambled classifiers
    def Parametros_AdaBoostClassifier(self):
        parameters = {}
        parameters['algorithm'] = ['SAMME', 'SAMME.R']
        parameters['n_estimators'] = [50, 100, 500]
        parameters['learning_rate'] = [0.01, 0.1, 1.0, 2.0]
        return parameters

    def Parametros_GradientBoostingClassifier(self):
        parameters = {}
        parameters['loss'] = ['deviance', 'exponential']
        parameters['learning_rate'] = [0.01, 0.1, 0.3]
        parameters['n_estimators'] = [100, 200, 300]
        parameters['max_depth'] = [3, 6, 9]
        parameters['criterion'] = ['friedman_mse', 'mse']
        parameters['max_features'] = [None, 'sqrt', 'log2']
        parameters['warm_start'] = [True, False]
        return parameters

    def Parametros_RandomForestClassifier(self):
        parameters = {}
        parameters['n_estimators'] = [5, 10, 15, 30]
        parameters['criterion'] = ['gini', 'entropy']
        parameters['max_features'] = [None, 'sqrt', 'log2']
        parameters['class_weight'] = ['balanced_subsample', None, 'balanced']
        parameters['max_depth'] = [2, 5, 10, 20]
        parameters['warm_start'] = [True, False]
        return parameters

    def Parametros_ExtraTreesClassifier(self):
        parameters = {}
        parameters['n_estimators'] = [10, 12, 15, 18, 20]
        parameters['criterion'] = ['gini', 'entropy']
        parameters['min_samples_leaf'] = [1, 2, 3, 4, 5]
        parameters['max_leaf_nodes'] = [3, 5, 7, 9, None]
        parameters['max_depth'] = [2, 3, 4, 5, None]
        parameters['max_features'] = [None, 'sqrt', 'log2']
        parameters['class_weight'] = ['balanced_subsample', None, 'balanced']
        return parameters

    def Parametros_VotingClassifier(self):
        parameters = {}
        parameters['voting'] = ['hard', 'soft']
        parameters['flatten_transform'] = [None, True, False]
        return parameters

    def Parametros_BaggingClassifier(self):
        parameters = {}
        parameters['n_estimators'] = [3, 6, 10]
        parameters['bootstrap'] = [True, False]
        parameters['bootstrap_features'] = [True, False]
        parameters['oob_score'] = [True, False]
        parameters['warm_start'] = [True, False]
        return parameters

    # regressors
    def Parametros_LinearRegression(self):
        parameters = {}
        parameters['normalize'] = [True, False]
        parameters['fit_intercept'] = [True, False]
        return parameters

    def Parametros_Lasso(self):
        parameters = {}
        parameters['fit_intercept'] = [False, True]
        parameters['normalize'] = [True, False]
        parameters['alpha'] = np.arange(0.1, 1.1, 0.3)
        parameters['selection'] = ['cyclic', 'random']
        parameters['tol'] = [0.001, 0.0001, 0.00001]
        return parameters

    def Parametros_Ridge(self):
        parameters = {}
        parameters['alpha'] = np.arange(0.1, 1.1, 0.3)
        parameters['fit_intercept'] = [True, False]
        parameters['normalize'] = [True, False]
        parameters['solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        return parameters

    def Parametros_ElasticNet(self):
        parameters = {}
        parameters['alpha'] = np.arange(0.1, 1.1, 0.3)
        parameters['l1_ratio'] = np.arange(0, 1.1, 0.2)
        parameters['fit_intercept'] = [True, False]
        parameters['normalize'] = [True, False]
        parameters['warm_start'] = [True, False]
        parameters['selection'] = ['cyclic', 'random']
        return parameters

    def Parametros_PassiveAggressiveRegressor(self):
        parameters = {}
        parameters['C'] = [0.8, 0.9, 1.0, 1.1, 1.2],
        parameters['fit_intercept'] = [True, False],
        parameters['warm_start'] = [True, False]
        return parameters

    def Parametros_SVR(self):
        parameters = {}
        parameters['C'] = [0.8, 0.9, 1.0, 1.1, 1.2]
        parameters['epsilon'] = [0.1, 0.001]
        parameters['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        parameters['shrinking'] = [True, False]
        parameters['C'] = [0.8, 0.9, 1.0, 1.1, 1.2]
        # parameters['degree'] = [1, 2, 3, 4] # depend just if poly
        parameters['max_iter'] = [10000]
        return parameters

    # emsambled regressors
    def Parametros_DecisionTreeRegressor(self):
        parameters = {}
        parameters['criterion'] = ['mse', 'friedman_mse']
        parameters['splitter'] = ['best', 'random']
        parameters['max_depth'] = [2, 3, 10, 50, 100]
        parameters['max_features'] = ['sqrt', 'log2', None]
        return parameters

    def Parametros_KNeighborsRegressor(self):
        parameters = {}
        parameters['n_neighbors'] = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50]
        parameters['weights'] = ['uniform', 'distance']
        parameters['algorithm'] = ['ball_tree', 'kd_tree', 'brute']
        parameters['p'] = [1, 2, 3, 4, 5]
        parameters['leaf_size'] = [10, 30, 50, 70]
        return parameters

    def Parametros_GaussianProcessRegressor(self):
        parameters = {}
        parameters['kernel'] = [None]  # try with kind of kernels
        parameters['alpha'] = np.logspace(-5, 0, 6)
        parameters['optimizer'] = ['fmin_l_bfgs_b']
        parameters['normalize_y'] = [True, False]
        return parameters

    def Parametros_AdaBoostRegressor(self):
        parameters = {}
        parameters['n_estimators'] = [50, 100, 500]
        parameters['learning_rate'] = [0.01, 0.1, 1.0, 2.0]
        parameters['loss'] = ['linear', 'square', 'exponential']
        return parameters

    def Parametros_GradientBoostingRegressor(self):
        parameters = {}
        parameters['loss'] = ['ls', 'lad', 'huber', 'quantile']
        parameters['learning_rate'] = [0.01, 0.1, 1.0, 2.0]
        parameters['n_estimators'] = [100, 200, 400]
        parameters['max_depth'] = [2, 5, 10, 20]
        parameters['criterion'] = ['mse', 'friedman_mse']
        parameters['max_features'] = ['sqrt', 'log2', None]
        # just for huber and quantile loss
        parameters['alpha'] = np.arange(0.1, 1.1, 0.3)
        parameters['warm_start'] = [True, False]
        return parameters

    def Parametros_RandomForestRegressor(self):
        parameters = {}
        parameters['n_estimators'] = [5, 10, 15, 20]
        parameters['criterion'] = ['mse', 'mae']
        parameters['max_features'] = [None, 'sqrt', 'log2']
        parameters['bootstrap'] = [True, False]
        parameters['oob_score'] = [True, False]
        parameters['warm_start'] = [True, False]
        return parameters

    def Parametros_ExtraTreesRegressor(self):
        parameters = {}
        parameters['n_estimators'] = [10, 12, 15, 18, 20]
        parameters['criterion'] = ['mse', 'mae']
        parameters['max_features'] = [None, 'sqrt', 'log2']
        parameters['bootstrap'] = [True, False] # REVISAR IT MUST BE TRUE
        parameters['oob_score'] = [True, False]
        parameters['warm_start'] = [True, False]
        #parameters['max_depth'] = [2, 3, 4, 5, None]
        return parameters

    def Parametros_BaggingRegressor(self):
        parameters = {}
        parameters['n_estimators'] = [5, 10, 15, 20]
        # parameters['max_samples'] = # int or float by choose
        # parameters['max_features'] = # int or float by choose
        parameters['bootstrap'] = [True, False]
        parameters['bootstrap_features'] = [True, False]
        parameters['oob_score'] = [True, False]
        parameters['warm_start'] = [True, False] # REVISAR IT MUST BE False
        return parameters

    def Heuristics_LogisticRegression(self): # 60
        return getHeuristicParamSearch(3, 15, 3, 0.5)
    
    def Heuristics_SGDClassifier(self): # 1440
        return getHeuristicParamSearch(10, 80, 5, 0.3)
    
    def Heuristics_PassiveAggressiveClassifier(self): # 80
        return getHeuristicParamSearch(4, 15, 3, 0.5)
    
    def Heuristics_MLPClassifier(self): # 720
        return getHeuristicParamSearch(6, 80, 3, 0.4)
    
    def Heuristics_LinearDiscriminantAnalysis(self): # 6
        return getHeuristicParamSearch(2, 3, 2, 0.5)
    
    def Heuristics_QuadraticDiscriminantAnalysis(self): # 4
        return getHeuristicParamSearch(2, 2, 2, 0.5)
    
    def Heuristics_KNeighborsClassifier(self): # 1440
        return getHeuristicParamSearch(10, 80, 5, 0.3)
    
    def Heuristics_DecisionTreeClassifier(self): # 120
        return getHeuristicParamSearch(4, 20, 3, 0.5)
    
    def Heuristics_GaussianNB(self): # 1
        return getHeuristicParamSearch(1, 1, 1, 0.5)
    
    def Heuristics_BernoulliNB(self): # 8
        return getHeuristicParamSearch(2, 3, 2, 0.5)
    
    def Heuristics_MultinomialNB(self): # 8
        return getHeuristicParamSearch(2, 3, 2, 0.5)
    
    def Heuristics_SVC(self, isPolinomic = True): # 160
        return getHeuristicParamSearch(6, 15, 3, 0.5)
    
    def Heuristics_AdaBoostClassifier(self): # 24
        return getHeuristicParamSearch(2, 10, 2, 0.5)
    
    def Heuristics_GradientBoostingClassifier(self): # 648
        return getHeuristicParamSearch(7, 30, 3, 0.4)
    
    def Heuristics_RandomForestClassifier(self): # 576
        return getHeuristicParamSearch(7, 50, 3, 0.4)
    
    def Heuristics_ExtraTreesClassifier(self): # 11250
        return getHeuristicParamSearch(12, 150, 10, 0.3)
    
    def Heuristics_VotingClassifier(self): # 6
        return getHeuristicParamSearch(2, 3, 2, 0.5)
    
    def Heuristics_BaggingClassifier(self): # 48
        return getHeuristicParamSearch(4, 10, 3, 0.5)
    
    def Heuristics_LinearRegression(self): # 4
        return getHeuristicParamSearch(2, 2, 2, 0.5)
    
    def Heuristics_Lasso(self): # 96
        return getHeuristicParamSearch(7, 12, 3, 0.5)
    
    def Heuristics_Ridge(self): # 96
        return getHeuristicParamSearch(7, 12, 3, 0.5)
    
    def Heuristics_ElasticNet(self): # 384
        return getHeuristicParamSearch(8, 30, 2, 0.5)
    
    def Heuristics_PassiveAggressiveRegressor(self): # 2
        return getHeuristicParamSearch(1, 2, 2, 0.5)
    
    def Heuristics_SVR(self): # 80
        return getHeuristicParamSearch(6, 10, 3, 0.5)
    
    def Heuristics_DecisionTreeRegressor(self): # 60
        return getHeuristicParamSearch(5, 10, 3, 0.5)
    
    def Heuristics_KNeighborsRegressor(self): # 1440
        return getHeuristicParamSearch(8, 50, 4, 0.4)
    
    def Heuristics_GaussianProcessRegressor(self): # 12
        return getHeuristicParamSearch(2, 5, 2, 0.5)
    
    def Heuristics_AdaBoostRegressor(self): # 36
        return getHeuristicParamSearch(3, 10, 2, 0.5)
    
    def Heuristics_GradientBoostingRegressor(self): # 9216
        return getHeuristicParamSearch(12, 120, 8, 0.3)
    
    def Heuristics_RandomForestRegressor(self): # 192
        return getHeuristicParamSearch(5, 25, 3, 0.5)
    
    def Heuristics_ExtraTreesRegressor(self): # 240
        return getHeuristicParamSearch(7, 30, 3, 0.5)
    
    def Heuristics_BaggingRegressor(self): # 64
        return getHeuristicParamSearch(5, 12, 3, 0.5)


def getHeuristicParamSearch(ngen, psize, elit, pelit):
    searchParams = {}
    searchParams['ngen'] = ngen # 3 # rand, eas, edas
    searchParams['psize'] = psize # 15 # eas, edas
    searchParams['elit'] = elit # 2 # eas
    searchParams['pelit'] = pelit # 0.5 # edas
    return searchParams

def parameterSpaceCounter(parameters):
    return reduce(lambda x,y: x*y, map(lambda x: len(parameters[x]), parameters.keys()))
