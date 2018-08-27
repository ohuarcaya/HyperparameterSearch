import pandas as pd
from Methods import getModelAccuracy
from edasSearch import EdasHyperparameterSearch
from Hiperparametros import HyperparameterSwitcher
from ImportacionModelos import getClassifierNames
from ImportacionModelos import getClassifierModels
from ImportacionModelos import getRegressorNames
from ImportacionModelos import getRegressorModels


if __name__ == "__main__":
    specificModelName = 'KNeighborsClassifier'
    estimadorDictionary = getClassifierModels(includeEnsambled=True)
    estimador = estimadorDictionary[specificModelName]
    hypSwitcher = HyperparameterSwitcher()
    parametros = hypSwitcher.getHyperparameters(specificModelName)()
    test = EdasHyperparameterSearch(getModelAccuracy, parametros, estimador,
                                    iterations=2, sample_size=15, select_ratio=0.3, debug=False)
    test.run()
    dataframe = pd.DataFrame(list(test.resultados)).sort_values(
        ['Accuracy'], ascending=False).reset_index(drop=True)
    dataframe.to_csv("results_test.csv")
