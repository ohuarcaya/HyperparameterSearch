"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import random as rnd
from collections import defaultdict

# Libreria Genetica
from deap import base, creator, tools, algorithms
# Multithread
from multiprocessing import Pool, Manager, cpu_count

# Subfunciones de estimadores
from sklearn.base import clone, BaseEstimator, TransformerMixin
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py][30]
from sklearn.base import is_classifier
from sklearn.utils import shuffle
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py][250]
from sklearn.utils.validation import _num_samples
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py][105]
from sklearn.utils.validation import indexable
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py][208]
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

# Selección para estimadores
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py][535]
from sklearn.model_selection._validation import _fit_and_score
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py][346]
from sklearn.model_selection._search import BaseSearchCV
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py][386]
from sklearn.model_selection._search import check_cv
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_split.py][1866]
from sklearn.model_selection._search import _check_param_grid

# Metricas para estimadores
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py][343]
from sklearn.metrics.scorer import check_scoring

#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings("ignore")
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from Methods import _createDataset
from Methods import _individual_to_params

seed = 7
test_size = 0.2
X, y = _createDataset("Tx_0x06")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)
X_train = np.int32(X_train)
X_test = np.int32(X_test)
y_train = np.int32(y_train)
y_test = np.int32(y_test)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.classification import accuracy_score
# estimador = SGDClassifier()

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
kernel = 1.0 * RBF([1.0, 1.0])
kernels = [1.0 * RBF(length_scale=1.0), 1.0 * DotProduct(sigma_0=1.0)**2]
estimador = GaussianProcessClassifier(kernels[1])

from sklearn.linear_model import PassiveAggressiveClassifier
estimador = PassiveAggressiveClassifier(max_iter=1000, tol=1e-6)

from sklearn.naive_bayes import MultinomialNB
alpha = 0.01
estimador = MultinomialNB()
X_trainTransf = X_train - np.min(np.min(X))
estimador.fit(X_trainTransf, y_train)
X_testTransf = X_test - np.min(np.min(X))
accuracy_score(y_test, estimador.predict(X_testTransf))


from sklearn.linear_model import Lasso
estimador = Lasso()
estimador.fit(X_train, y_train)
# regresion (y_test, estimador.predict(X_test))
from sklearn.linear_model import Ridge
estimador = Ridge()
estimador.fit(X_train, y_train)
# regresion (y_test, estimador.predict(X_test))
from sklearn.linear_model import ElasticNet
estimador = ElasticNet()
estimador.fit(X_train, y_train)
# regresion (y_test, estimador.predict(X_test))
from sklearn.linear_model import Perceptron
estimador = Perceptron()
estimador.fit(X_train, y_train)
np.mean(y_test == estimador.predict(X_test))
# accuracy_score(y_test, estimador.predict(X_test))
# from sklearn.linear_model import MultiTaskLasso
# estimador = MultiTaskLasso()
# estimador.fit(X_train, y_train)
# np.sum(y_test == estimador.predict(X_test))/len(y_test)
# accuracy_score(y_test, estimador.predict(X_test))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
estimador = QuadraticDiscriminantAnalysis()
estimador.fit(X_train, y_train)
np.mean(y_test == estimador.predict(X_test))
accuracy_score(y_test, estimador.predict(X_test))
from sklearn.naive_bayes import BernoulliNB
estimador = BernoulliNB()
estimador.fit(X_train, y_train)
np.mean(y_test == estimador.predict(X_test))
accuracy_score(y_test, estimador.predict(X_test))
from sklearn.neighbors import RadiusNeighborsClassifier
estimador = RadiusNeighborsClassifier(radius=1.0)
estimador.fit(X_train/10, y_train)
np.mean(y_test == estimador.predict(X_test/10))
accuracy_score(y_test, estimador.predict(X_test))

# sklearn.mixture.GaussianMixture
#http: // scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
from sklearn.mixture import GaussianMixture
n_classes = len(np.unique(y_train))
# Try GMMs using different types of covariances. , n_iter=20, init_params='wc', covariance_type='spherical'
estimador = GaussianMixture(n_components=15, max_iter=500, init_params='kmeans', tol=1e-6, covariance_type='spherical')
estimador.fit(X_train, y_train)
estimador.predict(X_test)
np.mean(y_test == estimador.predict(X_test))


from sklearn.ensemble import BaggingClassifier

estimador.fit(X_train, y_train)
accuracy_score(y_test, estimador.predict(X_test))

y_pred = estimador.predict(X_test)
y_pred = estimador.predict(X_test)
np.sum(y_pred == y_test)/len(y_test)

"""
LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
    class_weight=None, random_state=None, solver=’liblinear’, max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1)
LinearDiscriminantAnalysis(solver=’svd’, shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)
GaussianNB(priors=None)
MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, 
    learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, 
    tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
    class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’,
    metric_params=None, n_jobs=1, **kwargs)    
DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
    min_impurity_split=None, class_weight=None, presort=False)
RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
ExtraTreesClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
    bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3, min_impurity_decrease=0., min_impurity_split=None, 
    init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)
VotingClassifier(estimators, voting='hard', weights=None, n_jobs=1, flatten_transform=None)

# Pendientes
sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
sklearn.gaussian_process.GaussianProcessClassifier(kernel=None, optimizer=’fmin_l_bfgs_b’, n_restarts_optimizer=0, max_iter_predict=100, 
    warm_start=False, copy_X_train=True, random_state=None, multi_class=’one_vs_rest’, n_jobs=1)
sklearn.linear_model.SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 
    shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 
    class_weight=None, warm_start=False, average=False, n_iter=None)
sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001, store_covariances=None)
sklearn.linear_model.PassiveAggressiveClassifier(C=1.0, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, loss=’hinge’, 
    n_jobs=1, random_state=None, warm_start=False, class_weight=None, average=False, n_iter=None)
sklearn.linear_model.Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, 
    eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, n_iter=None)
sklearn.linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, 
    copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=’cyclic’)
sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, 
    warm_start=False, positive=False, random_state=None, selection=’cyclic’)
sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=’auto’, random_state=None)
sklearn.linear_model.MultiTaskLasso(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, 
    tol=0.0001, warm_start=False, random_state=None, selection=’cyclic’)
sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 
    bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
sklearn.neighbors.RadiusNeighborsClassifier(radius=1.0, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, 
    outlier_label=None, metric_params=None, **kwargs)

http://scikit-learn.org/stable/
http://scikit-learn.org/stable/modules/computational_performance.html#prediction-latency    

"""

"""
normalize = [True, False]
fit_intercept = [True, False]
copy_X = [True, False]
param_grid = dict(normalize=normalize,
                  fit_intercept=fit_intercept, copy_X=copy_X)
model = LinearRegression()
bestLR = tuneScaledModel(model, X_train, Y_train,
                         num_folds, seed, param_grid, scoring)
tuneResults(bestLR)
# 'LASSO'	Lasso()
normalize = [True, False]
alpha = [0.001, 0.01, 0.1]
selection = ['cyclic', 'random']
tol = [0.1, 0.3, 0.6]
param_grid = dict(normalize=normalize, alpha=alpha,
                  selection=selection, tol=tol)
model = Lasso()
bestLR = tuneScaledModel(model, X_train, Y_train,
                         num_folds, seed, param_grid, scoring)
tuneResults(bestLR)
# 'EN'	ElasticNet()
normalize = [True, False]
alpha = [0.001, 0.01, 0.1]
selection = ['cyclic', 'random']
tol = [0.01, 0.1, 0.2]
param_grid = dict(normalize=normalize, alpha=alpha,
                  selection=selection, tol=tol)
model = ElasticNet()
bestLR = tuneScaledModel(model, X_train, Y_train,
                         num_folds, seed, param_grid, scoring)
tuneResults(bestLR)
# 'KNN'	KNeighborsRegressor()
# Tune scaled KNN Reg
n_neighbors = [1, 3, 5, 7, 9]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid = dict(n_neighbors=n_neighbors,
                  weights=weights, algorithm=algorithm)
model = KNeighborsRegressor(n_jobs=-1)
bestKNN = tuneScaledModel(model, X_train, Y_train,
                          num_folds, seed, param_grid, scoring)
tuneResults(bestKNN)
# 'CART'	DecisionTreeRegressor()
# Tune scaled CART Reg
criterion = ['mse']  # 'mae',
max_features = [None, 'sqrt', 'log2']
max_depth = [5, 10, 15, 20]
splitter = ["random", "best"]
param_grid = dict(criterion=criterion, max_features=max_features,
                  splitter=splitter, max_depth=max_depth)
model = DecisionTreeRegressor()
bestLR = tuneScaledModel(model, X_train, Y_train,
                         num_folds, seed, param_grid, scoring)
tuneResults(bestLR)
# 'SVR'	SVR()
C = [0.1, 1, 10]  # 'mae',
epsilon = [0.1, 0.5, 1.0]
max_depth = [5, 10, 15, 20]
kernel = ["rbf", "linear", "poly", "sigmoid"]
degree = [2, 3, 4, 5]
param_grid = dict(C=C, epsilon=epsilon, kernel=kernel, degree=degree)
model = SVR()
bestLR = tuneScaledModel(model, X_train, Y_train,
                         num_folds, seed, param_grid, scoring)
tuneResults(bestLR)
# 'AB'	AdaBoostRegressor()
n_estimators = [500, 1000]
learning_rate = [0.1, 0.5, 1]
loss = ['linear', 'square', 'exponential']
param_grid = dict(n_estimators=n_estimators,
                  learning_rate=learning_rate, loss=loss)
model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3))
bestLR = tuneScaledModel(model, X_train, Y_train,
                         num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

"""
