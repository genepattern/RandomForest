# Random Forest

**Description**: The following is a GenePattern module written in Python 3. It performs [random forest classification](/docs/randomforest.md) by either <ins>cross-validation</ins> (takes one dataset as input, done through LOOCV, [leave-one-out cross validation](/docs/randomforest.md#leave-one-out-cross-validation)) or <ins>test-train prediction</ins> (takes two datasets, test and train). Each dataset consists of two file inputs, one for feature data (.gct), and one for target data (.cls). It uses Scikit-learn's RandomForestClassifier (v1.2). Also includes several optional parameters for specifying the classification algorithm process.

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Summary

This repository is a GenePattern module written in [Python 3](https://www.python.org/download/releases/3.0/).


It processes files into DataFrames and performs random forest classification (uses LOOCV (leave-one-out cross validation) in the case of cross-validation) on them using Scikit-learn's RandomForestClassifier, generating an accuracy score and a prediction results file (.pred.odf) that compares the "true" class to the model's prediction. Created for module usage through optional arguments for classifier parameters.


## Source Links
* [The GenePattern RandomForest source repository](/../../)
* RandomForest uses the [genepattern/randomforest:0.3](https://hub.docker.com/layers/genepattern/randomforest/0.3/images/sha256-c8568e0bcc7740b95d19065a47cde1a40df8c436b8ca8eb69892f5c56b02e0d2?context=explore) docker image

## Usage
For <ins>cross-validation</ins>, the module only requires one feature data file (.gct) and one target  data file (.cls). For <ins>test-train prediction</ins>, the module **additionally** requires another dataset in the form of a testing feature (.gct) and testing target (.cls) data file. Other parameters for classifier specifications are optional, maintaining default values if left unchanged (see below).

## Parameters

| Name | Description | Default Value | Cross-Validation | Test-Train Prediction
---------|--------------|----------------|----------------|----------------
| train.data.filename* |  Required feature data file to be read from user (.gct) | No default value | ✔ | ✔ |
| train.class.filename* |  Required target data file to be read from user (.cls) | No default value | ✔ | ✔ |
| test.data.filename |  Optional (only provide when doing test-train prediction) testing feature data file to be read from user (.gct) | No default value |  | ✔ |
| test.class.filename |  Optional (only provide when doing test-train prediction) testing target data file to be read from user (.cls) | No default value |  | ✔ |
| prediction.results.filename* | Optional prediction results filename (.pred.odf, follows [GP ODF format](https://www.genepattern.org/file-formats-guide#ODF)) | (train.data.file_basename).pred.odf |
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
| oob_score | Optional boolean for if out-of-bag samples used for generalization score; if bootstrap is False, can only be False | False |
| random_state | Optional int for seed of random number generator (nonnegative, caps at 4294967295, 2<sup>32</sup> - 1), also takes None. Note: Setting this to a specific integer, like 0 for example, for a specific dataset, will always yield the same prediction results file as this argument controls how bagging and random feature selection for a specific dataset occur.| None |
| debug | Optional boolean for program debugging | False |
| verbose | Optional int (0 = no verbose, 1 = base verbosity) to increase classifier verbosity (non-negative), [more info](https://scikit-learn.org/stable/glossary.html#term-verbose) (for other input values) | 0 |

\*  required

## Input Files

1. Training Data Feature File   
    This is the required input file of classifier training feature data which is used to create the random forest model. For cross-validation, this is the only feature data input which also has prediction done against it via LOOCV. The parameter expects a GCT file (.gct) that follows the [GenePattern GCT](https://www.genepattern.org/file-formats-guide#GCT) file standard.
      
2. Training Data Class File   
    This is the required input file of classifier training target data which is used to create the random forest model. For cross-validation, this is the only target data input whose values are considered as "true." The parameter expects a CLS file (.cls) that follows the [GenePattern CLS](https://www.genepattern.org/file-formats-guide#CLS) file standard.

3. Testing Data Feature File   
    This is the optional (only passed in for test-train prediction) input file of classifier testing feature data which the random forest model will predict the class values of. The parameter expects a GCT file (.gct) that follows the [GenePattern GCT](https://www.genepattern.org/file-formats-guide#GCT) file standard.
      
4. Testing Data Class File   
    This is the optional (only passed in for test-train prediction) input file of classifier testing target data whose values are considered as "true." The parameter expects a CLS file (.cls) that follows the [GenePattern CLS](https://www.genepattern.org/file-formats-guide#CLS) file standard.
    
## Output Files

Outputs a results file (.pred.odf) file that follows the [GenePattern ODF (Output Description Format)](https://www.genepattern.org/file-formats-guide#ODF) file standard. It contains a specific set of descriptive headers followed by a main data block comparing the random forest classification's predictions on the entire feature dataset against the true values.


## Test-Train Example Data

ALL_AML Dataset Inputs:
[all_aml_train.gct](/data/all_aml_train.gct), [all_aml_train.cls](/data/all_aml_train.cls), [all_aml_test.gct](/data/all_aml_train.gct), and [all_aml_test.cls](/data/all_aml_train.cls)  
ALL_AML Example Output:
[all_aml_tt.pred.odf](/data/example_output/all_aml_tt.pred.odf)


## Cross-Validation Example Data

ALL_AML Dataset Inputs:
[all_aml_train.gct](/data/all_aml_train.gct) and [all_aml_train.cls](/data/all_aml_train.cls)  
ALL_AML Example Output:
[all_aml_xval.pred.odf](/data/example_output/all_aml_xval.pred.odf)

BRCA_HUGO Dataset Inputs:
[DP_4_BRCA_HUGO_symbols.preprocessed.gct](/data/DP_4_BRCA_HUGO_symbols.preprocessed.gct) and [Pred_2_BRCA_HUGO_symbols.preprocessed.cls](/data/Pred_2_BRCA_HUGO_symbols.preprocessed.cls)  
BRCA_HUGO Example Output:
[DP_4_BRCA_HUGO_symbols.preprocessed.pred.odf](/data/example_output/DP_4_BRCA_HUGO_symbols.preprocessed.pred.odf)

Iris Dataset Inputs:
[iris.gct](/data/iris.gct) and [iris.cls](/data/iris.cls)  
Iris Example Output:
[iris.pred.odf](/data/example_output/iris.pred.odf)

## Requirements

Requires the [genepattern/randomforest:0.3](https://hub.docker.com/layers/genepattern/randomforest/0.3/images/sha256-c8568e0bcc7740b95d19065a47cde1a40df8c436b8ca8eb69892f5c56b02e0d2?context=explore) docker image

## Miscellaneous

Future development ideas:
* Handling the following miscellaneous input arguments: class_weight input of dictionary/list of; max_features input of int/float


## License

`RandomForest` is distributed under a modified BSD license available [here](https://github.com/genepattern/RandomForestClassifier/blob/main/LICENSE.txt)

## Version Comments

| Version | Release Date | Description                                 |
----------|--------------|---------------------------------------------|
| 1.0.0 | Jan 2, 2023 | Initial version for team use. |
