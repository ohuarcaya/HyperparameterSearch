# Hyperparameter Search Optimization

Hyperparameter search optimization by genetic algorithms and distribution estimation algorithms

## Virtual Environment
```sh
python3 -m venv env
source env/bin/activate
```

## About this work

- __Types of Hyperparameter Search to Compare__

        - EAS
        - EDAS
        - GridSearch
        - Randomized

- __Metrics to Compare__

        - Accuracy, Precision
        - Runtime Execution
        - Number of Iterations
        - Computational consumption (CPU, Memory, Energy)

- __Cases of Study__

        - Classification problem With Indoor Location
        - Regression problem With Indoor Location

- __Details__

        - Evaluation of changes by univariate and bivariate EAS, EDAS.
        - Dataset Wlan dataset for indoor localisation: 

[aquí el enlace uci](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc#)

- __Benchmark study__

        Librerías para realizar benchmark en python

                - pycallgraph
                - cProfile
                - line_profiler
                - memory_profiler
                - timeit
                - profilehooks
		- guppy

        Enlaces de Referencia

                - https://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/
                - https://github.com/ionelmc/pytest-benchmark
                - https://mg.pov.lt/profilehooks/
                - https://pypi.org/project/pytest-benchmark/
                - https://stackoverflow.com/questions/1593019/is-there-any-simple-way-to-benchmark-python-script
                - https://dzone.com/articles/unit-testing-the-good-bad-amp-ugly
		- http://www.marinamele.com/7-tips-to-time-python-scripts-and-control-memory-and-cpu-usage
