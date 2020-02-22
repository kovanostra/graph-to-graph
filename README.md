![Graph to graph](https://github.com/kovanostra/graph-to-graph/workflows/Graph%20to%20graph/badge.svg)

# Multi-modal Graph-to-Graph Translation for Molecular Optimization

### Overview

This project aims at rewriting the code found in https://github.com/wengong-jin/iclr19-graph2graph, which corresponds to Jin et al. ICLR 2019 (https://arxiv.org/abs/1812.01070)

The goal is to obtain a repository for educational purposes with:
1. the same functionality, 
2. little or no dependency on external machine learning libraries,
3. no gpu dependency,
4. well tested code, and 
5. more readable code.

It is not a goal of this project to provide a better performing or faster model.

### Requirements
Python 3.6 

Run (only for tests at the moment)
```
numpy==1.17.4
rdkit==2019.09.3.0
```

Rdkit can be installed through conda: https://anaconda.org/rdkit/rdkit

Build
```
tox==3.14.3
tox-conda==0.2.1
```

tox-conda can be installed through pip and it is necessary for building when using conda packages: https://github.com/tox-dev/tox-conda


To run all tests and build the project, just cd to ~/graph_to_graph/ and run (with sudo if necessary)
```
tox
```

This will automatically create an artifact and place it in ~/graph_to_graph/.tox/dist/graph-to-graph-version.zip. The version can be specified in the setup.py. The contents of this folder are cleaned at the start of every new build.

### Entrypoint

Currently, there is no entrypoint. However, the code can be explored through the tests.


### Progress

-[x] Graph encoder \
-[x] Tree decomposition \
-[x] Junction tree encoder \
-[ ] Variational autoencoder \
-[ ] Junction tree decoder \
-[ ] Graph decoder \
-[ ] Training \
-[ ] Hyperparameter optimization \
-[ ] Evaluation 

### Azure pipelines project

https://dev.azure.com/kovamos/graph-to-graph
