import pandas as pd
import sys
sys.path.append("../lib/")
sys.path.append("../data/")
from Methods import GeneralMethods
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
    gm = GeneralMethods(estimador, urlDataset='../data/Tx_0x06')
    test = EdasHyperparameterSearch(
        gm, parametros, estimador, iterations=2, sample_size=15, select_ratio=0.3, debug=False)
    test.run()
    dataframe = pd.DataFrame(list(test.resultados)).sort_values(
        ['Accuracy'], ascending=False).reset_index(drop=True)
    dataframe.to_csv("../result/result_" + specificModelName + ".csv")
