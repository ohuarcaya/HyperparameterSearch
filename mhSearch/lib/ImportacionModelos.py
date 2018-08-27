import warnings
warnings.filterwarnings("ignore")
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
# Ensemble Classifiers
from sklearn.ensemble import AdaBoostClassifier # Future deprecated dependency (numpy.core.umath_tests)
from sklearn.ensemble import GradientBoostingClassifier # Future deprecated dependency(numpy.core.umath_tests)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
# Regresors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
# Ensemble Regresors
from sklearn.ensemble import AdaBoostRegressor # Future deprecated dependency(numpy.core.umath_tests)
from sklearn.ensemble import GradientBoostingRegressor # Future deprecated dependency(numpy.core.umath_tests)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
# from sklearn.linear_model import Perceptron
# Perceptron and SGDClassifier share the same underlying implementation. In fact, Perceptron() is equivalent to SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
# from sklearn.neighbors import RadiusNeighborsClassifier # TO MUCH DANGEROUS BECAUSE OF RADIOUS PRODUCE EXCEPTION

def getClassifierNames(includeEnsambled=False):
    names = ['LogisticRegression', 'SGDClassifier', 'PassiveAggressiveClassifier', 'MLPClassifier', 
            'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'KNeighborsClassifier', 
            'DecisionTreeClassifier', 'GaussianNB', 'BernoulliNB', 'MultinomialNB', 'SVC']
    if includeEnsambled:
        names = names + ['AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier',
             'ExtraTreesClassifier', 'VotingClassifier', 'BaggingClassifier']
    return names


def getRegressorNames(includeEnsambled=False):
    names = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'PassiveAggressiveRegressor', 'SVR', 
            'DecisionTreeRegressor', 'KNeighborsRegressor', 'GaussianProcessRegressor']
    if includeEnsambled:        
        names = names + ['AdaBoostRegressor', 'GradientBoostingRegressor', 'RandomForestRegressor', 
        'ExtraTreesRegressor', 'BaggingRegressor']
    return names

def getModelNameAbreviation(modelName):
    abreviatura = {}
    abreviatura['LogisticRegression'] = 'LoR'
    abreviatura['SGDClassifier'] = 'SGD'
    abreviatura['PassiveAggressiveClassifier'] = 'PA'
    abreviatura['MLPClassifier'] = 'MLP'
    abreviatura['LinearDiscriminantAnalysis'] = 'LDA'
    abreviatura['QuadraticDiscriminantAnalysis'] = 'QDA'
    abreviatura['KNeighborsClassifier'] = 'KNN'
    abreviatura['DecisionTreeClassifier'] = 'DT'
    abreviatura['GaussianNB'] = 'GNB'
    abreviatura['BernoulliNB'] = 'BNB'
    abreviatura['MultinomialNB'] = 'MNB'
    abreviatura['SVC'] = 'SVM'
    abreviatura['AdaBoostClassifier'] = 'AB'
    abreviatura['GradientBoostingClassifier'] = 'GBM'
    abreviatura['RandomForestClassifier'] = 'RF'
    abreviatura['ExtraTreesClassifier'] = 'ET'
    abreviatura['VotingClassifier'] = 'VoC'
    abreviatura['BaggingClassifier'] = 'BAG'
    abreviatura['LinearRegression'] = 'LiR'
    abreviatura['Lasso'] = 'LaR'
    abreviatura['Ridge'] = 'RiR'
    abreviatura['ElasticNet'] = 'EN'
    abreviatura['PassiveAggressiveRegressor'] = 'PA'
    abreviatura['SVR'] = 'SVM'
    abreviatura['DecisionTreeRegressor'] = 'DT'
    abreviatura['KNeighborsRegressor'] = 'KNN'
    abreviatura['GaussianProcessRegressor'] = 'GBM'
    abreviatura['AdaBoostRegressor'] = 'AB'
    abreviatura['GradientBoostingRegressor'] = 'GBM'
    abreviatura['RandomForestRegressor'] = 'RF'
    abreviatura['ExtraTreesRegressor'] = 'ET'
    abreviatura['BaggingRegressor'] = 'BAG'
    return abreviatura[modelName]

def getClassifierModels(includeEnsambled=False, seed=7):
    models = {}
    models['LogisticRegression'] = LogisticRegression()
    models['SGDClassifier'] = SGDClassifier(random_state=seed)
    models['PassiveAggressiveClassifier'] = PassiveAggressiveClassifier(random_state=seed)
    models['MLPClassifier'] = MLPClassifier(random_state=seed)
    models['LinearDiscriminantAnalysis'] = LinearDiscriminantAnalysis()
    models['QuadraticDiscriminantAnalysis'] = QuadraticDiscriminantAnalysis()
    models['KNeighborsClassifier'] = KNeighborsClassifier()
    models['DecisionTreeClassifier'] = DecisionTreeClassifier(random_state=seed)
    models['GaussianNB'] = GaussianNB()
    models['BernoulliNB'] = BernoulliNB()
    models['MultinomialNB'] = MultinomialNB()
    models['SVC'] = SVC(random_state=seed)
    if(includeEnsambled):
        models['AdaBoostClassifier'] = AdaBoostClassifier(DecisionTreeClassifier(random_state=seed), random_state=seed)
        models['GradientBoostingClassifier'] = GradientBoostingClassifier(random_state=seed)
        models['RandomForestClassifier'] = RandomForestClassifier(random_state=seed)
        models['ExtraTreesClassifier'] = ExtraTreesClassifier(random_state=seed)
        estimators = []
        estimators.append(("Voting_GradientBoostingClassifier", GradientBoostingClassifier(random_state=seed)))
        estimators.append(("Voting_ExtraTreesClassifier", ExtraTreesClassifier(random_state=seed)))
        models['VotingClassifier'] = VotingClassifier(estimators)
        models['BaggingClassifier'] = BaggingClassifier(DecisionTreeClassifier(random_state=seed), random_state=seed)
    return models


def getRegressorModels(includeEnsambled=False, seed=7):
    models = {}
    models['LinearRegression'] = LinearRegression()
    models['Lasso'] = Lasso(random_state=seed)
    models['Ridge'] = Ridge(random_state=seed)
    models['ElasticNet'] = ElasticNet(random_state=seed)
    models['PassiveAggressiveRegressor'] = PassiveAggressiveRegressor(random_state=seed)
    models['SVR'] = SVR()
    models['DecisionTreeRegressor'] = DecisionTreeRegressor(random_state=seed)
    models['KNeighborsRegressor'] = KNeighborsRegressor()
    models['GaussianProcessRegressor'] = GaussianProcessRegressor(random_state=seed)
    if(includeEnsambled):		# Regression
        models['AdaBoostRegressor'] = AdaBoostRegressor(DecisionTreeRegressor(random_state=seed), random_state=seed)
        models['GradientBoostingRegressor'] = GradientBoostingRegressor(random_state=seed)
        models['RandomForestRegressor'] = RandomForestRegressor(random_state=seed)
        models['ExtraTreesRegressor'] = ExtraTreesRegressor(random_state=seed)
        models['BaggingRegressor'] = BaggingRegressor(DecisionTreeRegressor(random_state=seed), random_state=seed)
    return models

