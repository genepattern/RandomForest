# What is Random Forest?

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Introduction

Random Forest is a an [_ensemble_](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/) machine learning algorithm. Specifically, it is an ensemble of **decision trees** (hence the _forest_ part) which basically means that it combines multiple decsision trees in its model. In order to understand Random Forest classification and regression, let us first take one step back and understand what decision trees are.

## Decision Trees

Decision trees are fundamentally quite simple, as they essentially help you answer a classification question (i.e, "Is it X or Y"). These "X" and "Y" (and however many you want) labels are called **target** values. In classification, they are essentially what we want s is best illustrated through an example. In technical terms, it is a ""

_Note_: Decision trees by no means have to be binary trees. They could be ternary (one decision node splits into 3 leaves), or they could have as many splits as you want. It all depends on the conditional statement(s) for each decision node. In fact, any ternary decision tree could be represented as a decision tree, and that applies to any ([this answer is a nice sanity check for that](https://stats.stackexchange.com/a/12227)). **HOWEVER**, the reason you may almost always see decision trees as binary is probably due to two asepcts.

1. asdfjldkafs
2. 


## Decision Tree Example
* [The GenePattern RandomForest source repository](https://github.com/omarhalawa3301/randomforest)
* RandomForest uses the [omarhalawa/randomforest:1.0](https://hub.docker.com/layers/omarhalawa/randomforest/1.0/images/sha256-995d424aa0fa77f608aaa5575faafad6cea966a377fdb8dd51e9144e74f7ff21?context=repo) docker image

## Motivation
This module only requires feature (.gct) and target (.cls) classifier data files as well as an output filename as user-input. Other parameters are optional, maintaining default values if left unchanged (see below).

## Parameters

| Name | Description | Default Value |
---------|--------------|----------------
| data.file * |  Classifier feature data filename to be read from user (.gct, more format support to come) | No default value |
| cls.file * |  Classifier target data filename to be read from user (.cls, more format support to come) | No default value |
| prediction.results.filename * |  Classifier prediction results filename (.pred.odf, follows [GP ODF format](https://www.genepattern.org/file-formats-guide#ODF)) | (data.file_basename).pred.odf |
| bootstrap | Optional boolean to turn on classifier bootstrapping | True |
| ccp_alpha | Optional float for complexity parameter of min cost-complexity pruning (>= 0.0) | 0.0 |
| class_weight | Optional string for class weight specification of either of: {"balanced," "balanced_subsample"}, also takes None ; (**future implementation:** to handle input of dictionary/list of); Note: "balanced" or "balanced_subsample" are not recommended for warm start if the fitted data differs from the full dataset | None |
| criterion | Optional string for node-splitting criterion of one of the following: {“gini”, “entropy”, “log_loss”} | "gini" |
| max_depth | Optional int for maximum tree depth (>= 1), also takes None | None |
| max_features | Optional string for number of features per split of either one of the following: {"sqrt," "log2"} ("auto" to be removed in Scikit 1.3), (**future implementation:** handle input of float/int) | "sqrt" |
| max_leaf_nodes | Optional int for maximum leaf nodes per tree (>= 2), also takes None | None |
| max_samples | Optional float for ratio of datasets to use per tree (between 0.0 and 1.0, inclusive for both), also takes None; if bootstrap is False, can only be None | None |
| min_impurity_decrease | Optional float for minimum impurity decrease needed per node split (>= 0.0) | 0.0 |
| min_samples_leaf | Optional int for minimum number of samples required at leaf node (>= 1) | 1 |
| min_samples_split | Optional int for minimum sample number to split node (>= 2) | 2 |
| min_weight_fraction_leaf | Optional float for min weighted fraction of weight sum total to be leaf (between 0.0 and 0.5, inclusive for both) | 0.0 |
| n_estimators | Optional int for number of trees in forest (>= 1) | 100 |
| n_jobs | Optional int for number of parallel streams for building the forest (nonzero), also takes None, [more info](https://scikit-learn.org/stable/glossary.html#term-n_jobs) (-1 for all CPUs) | None |
| oob_score | Optional boolean for if out-of-bag samples used for generalization score; if bootstrap is False, can only be False | False |
| random_state | Optional int for seed of random number generator (nonnegative, caps at 4294967295, 2<sup>32</sup> - 1), also takes None | None |
| warm_start | Optional boolean for whether to start new forest or add to past solution | False |
| debug | Optional boolean for program debugging | False |
| verbose | Optional int (0 = no verbose, 1 = base verbosity) to increase classifier verbosity (non-negative), [more info](https://scikit-learn.org/stable/glossary.html#term-verbose) (for other input values) | 0 |

\*  required

## Required Inputs

1. data file  
    This is the input file of classifier feature data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a GCT file (.gct), but future support for other feature data formats will be implemented.  
      
2. cls file  
    This is the input file of classifier target data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a CLS file (.cls), but future support for other feature data formats will be implemented.  

## Miscellaneous

Future development ideas:
* Further prediction on other user-input feature data using results
* .res file implementation for feature data
* .txt file implementation for both feature and target data
* Handling the following miscellaneous input arguments: class_weight input of dictionary/list of; max_features input of int/float
