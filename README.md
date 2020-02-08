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
```

Build
```
tox==3.14.3
```
To run all tests and build the project, just cd to Graph_to_graph/ and run (with sudo if necessary)
```
tox
```

### Entrypoint

Currently, there is no entrypoint. However, the code can be explored through the tests.


### Progress

-[x] Graph encoder \
-[ ] Graph to Junction tree encoder \
-[x] Junction tree encoder \
-[ ] Variational autoencoder \
-[ ] Junction tree decoder \
-[ ] Graph decoder \
-[ ] Training \
-[ ] Hyperparameter optimization \
-[ ] Evaluation 

### Azure pipelines project

https://dev.azure.com/kovamos/graph-to-graph
