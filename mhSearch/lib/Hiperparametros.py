import numpy as np


class HyperparameterSwitcher(object):
    def getHyperparameters(self, modelName):
        methodName = 'Parametros_' + str(modelName)
        method = getattr(self, methodName, lambda: "Invalid Model Name")
        return method

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
        parameters['loss'] = ['hinge', 'log',
                              'modified_huber', 'squared_hinge', 'perceptron']
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

    def Parametros_DecisionTreeClassifier(self, isDummy = True):
        parameters = {}
        if isDummy:
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

    def Parametros_SVC(self, isPolinomic = True):
        parameters = {}
        parameters['C'] = [1, 2, 5, 10]
        parameters['decision_function_shape'] = ['ovr']
        parameters['max_iter'] = [10000]
        parameters['shrinking'] = [True, False]
        if isPolinomic:
            parameters['kernel'] = ['poly']
            parameters['degree'] = [1, 2, 3, 4, 5] # conditional for kernel poly
        else:
            parameters['kernel'] = ['linear', 'rbf', 'sigmoid']
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
        parameters['solver'] = ['svd', 'cholesky',
                                'lsqr', 'sparse_cg', 'sag', 'saga']
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
        parameters['bootstrap'] = [True, False]
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
        parameters['warm_start'] = [True, False]
        return parameters
