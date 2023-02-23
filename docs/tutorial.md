# RandomForest (Non-GPU)

**Description**: The following is a GenePattern module written in Python 3. It performs random forest classification via LOOCV ([leave-one-out cross validation](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)) on feature and target data files and outputs prediction results. It uses Scikit-learn's RandomForestClassifier (v1.2). Also includes several optional parameters for specifying the classification algorithm process. This module/repo serves as a foundation for implementing the cuML-based GPU Random Forest Classifier.

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Summary

This repository is a GenePattern module written in [Python 3](https://www.python.org/download/releases/3.0/).


It takes in two files, one for feature data (.gct), and one for target data (.cls). Then, it processes them into DataFrames and performs random forest classification via LOOCV (leave-one-out cross validation) on them using Scikit-learn's RandomForestClassifier, generating an accuracy score and a target prediction done on the entire feature dataset. Outputs a prediction results (.pred.odf) file. Created for module usage through optional arguments for classifier parameters.


## Source Links
* [The GenePattern RandomForest source repository](https://github.com/omarhalawa3301/randomforest)
* RandomForest uses the [omarhalawa/randomforest:1.0](https://hub.docker.com/layers/omarhalawa/randomforest/1.0/images/sha256-995d424aa0fa77f608aaa5575faafad6cea966a377fdb8dd51e9144e74f7ff21?context=repo) docker image

## Usage
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

## Input Files

1. data file  
    This is the input file of classifier feature data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a GCT file (.gct), but future support for other feature data formats will be implemented.  
      
2. cls file  
    This is the input file of classifier target data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a CLS file (.cls), but future support for other feature data formats will be implemented.  

    
## Output Files

Outputs a results file (.pred.odf) file that follows the [GenePattern ODF (Open Document Format)](https://www.genepattern.org/file-formats-guide#ODF) file standard. It contains a specific set of descriptive headers followed by a main data block comparing the random forest classification's predictions on the entire feature dataset against the true values.

## Example Data

ALL_AML Dataset Inputs:
[all_aml_train.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.gct) and [all_aml_train.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.cls)  
ALL_AML Example Output:
[all_aml.pred.odf](https://github.com/omarhalawa3301/randomforest/blob/main/data/example_output/all_aml.pred.odf)

BRCA_HUGO Dataset Inputs:
[DP_4_BRCA_HUGO_symbols.preprocessed.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/DP_4_BRCA_HUGO_symbols.preprocessed.gct) and [Pred_2_BRCA_HUGO_symbols.preprocessed.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/Pred_2_BRCA_HUGO_symbols.preprocessed.cls)  
Iris Example Output:
[DP_4_BRCA_HUGO_symbols.preprocessed.pred.odf](https://github.com/omarhalawa3301/randomforest/blob/main/data/example_output/DP_4_BRCA_HUGO_symbols.preprocessed.pred.odf)

Iris Dataset Inputs:
[iris.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.gct) and [iris.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.cls)  
Iris Example Output:
[iris.pred.odf](https://github.com/omarhalawa3301/randomforest/blob/main/data/example_output/iris.pred.odf)

## Requirements

Requires the [omarhalawa/randomforest:1.0](https://hub.docker.com/layers/omarhalawa/randomforest/1.0/images/sha256-995d424aa0fa77f608aaa5575faafad6cea966a377fdb8dd51e9144e74f7ff21?context=repo) docker image

## Miscellaneous

Future development ideas:
* Further prediction on other user-input feature data using results
* .res file implementation for feature data
* .txt file implementation for both feature and target data
* Handling the following miscellaneous input arguments: class_weight input of dictionary/list of; max_features input of int/float


## License

`RandomForest` is distributed under a modified BSD license available [here](https://github.com/omarhalawa3301/randomforest/blob/main/LICENSE.txt)

## Version Comments

| Version | Release Date | Description                                 |
----------|--------------|---------------------------------------------|
| 1.0.0 | Jan 2, 2023 | Initial version for team use. |
