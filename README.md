# Hyperparameter Search Optimization for Machine Learning Models, focused in indoor location

Hyperparameter search comparison of different heuristics in order to optimize the time spend in find
the best configuration for Machine Learning Models, the study case takes a multi building, multi floor 
indoor localization database to test Indoor Positioning System that rely on WLAN/WiFi fingerprint.

[aquí el enlace uci](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc#)


## About this work

- __Details__

        - Dataset with plenty of wifi signal for indoor localisation in 3 buildings
        - Indoor localization 3D: latitude, longitude, floor

- __Cases of Study__

        - Classification to predict building and floor
        - Regression to predict latitude and longitude

- __Types of Hyperparameter Search to Compare__

        - EAS (Evolutive Algorithm Search)
        - EDAS (Estimation of Distributions Algorithm Search)
        - GridSearch (Exhaustive Search)
        - RandomizedSearch (Randomized Search)

- __Metrics to Compare__

        - Accuracy with Classifiers
        - Mse for distance error with Regressors
        - Runtime Execution
        - Number of Iterations
        - Computational consumption (CPU, RAM, Energy, Cycles)

- __new changes__

        - AdaBoostRegressorLONGITUDE.csv = 36
        - BaggingRegressorLATITUDE.csv = 101
        - ExtraTreesRegressorLATITUDE.csv = 673
        - KNeighborsRegressorLONGITUDE.csv = 1441
        - RandomForestRegressorLATITUDE.csv = 481