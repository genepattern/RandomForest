# Random Forest

**Description**: The following is a GenePattern module written in Python 3. It performs [random forest classification](/docs/randomforest.md) by either <ins>cross-validation</ins> (takes one dataset as input, done through LOOCV, [leave-one-out cross validation](/docs/randomforest.md#leave-one-out-cross-validation)) or <ins>test-train prediction</ins> (takes two datasets, test and train). Each dataset consists of two file inputs, one for feature data (.gct), and one for target data (.cls). It uses RAPIDS.ai's RandomForestClassifier (cuML 23.08.00) and sends jobs to the [San Diego Supercomputer Center](https://www.sdsc.edu/). Also includes several optional parameters for specifying the classification algorithm process.

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Summary

This repository is a GenePattern module written in [Python 3](https://www.python.org/download/releases/3.0/).


It processes files into DataFrames and performs random forest classification (uses LOOCV (leave-one-out cross validation) in the case of cross-validation) on them using Scikit-learn's RandomForestClassifier, generating an accuracy score and a prediction results file (.pred.odf) that compares the "true" class to the model's prediction. Created for module usage through optional arguments for classifier parameters.


## Source Links
* [The GenePattern RandomForest source repository](/../../)
* RandomForest uses the following NMF-GPU Singularity container found [here](https://github.com/genepattern/nmf-gpu/blob/master/docker/Singularity.def) (created from Dockerfile).

## Usage
For <ins>cross-validation</ins>, the module only requires one feature data file (.gct) and one target  data file (.cls). For <ins>test-train prediction</ins>, the module **additionally** requires another dataset in the form of a testing feature (.gct) and testing target (.cls) data file. Other parameters for classifier specifications are optional, maintaining default values if left unchanged (see below).

## Parameters

| Name | Description | Default Value | Cross-Validation | Test-Train Prediction
---------|--------------|----------------|----------------|----------------
| train.data.file* |  Required feature data file to be read from user (.gct) | No default value | ✔ | ✔ |
| train.class.file* |  Required target data file to be read from user (.cls) | No default value | ✔ | ✔ |
| test.data.file |  Optional (only provide when doing test-train prediction) testing feature data file to be read from user (.gct) | No default value |  | ✔ |
| test.class.file |  Optional (only provide when doing test-train prediction) testing target data file to be read from user (.cls) | No default value |  | ✔ |
| prediction.results.filename* | Optional prediction results filename (.pred.odf, follows [GP ODF format](https://www.genepattern.org/file-formats-guide#ODF)); also applied to optional model output basenames | (train.data.file_basename).pred.odf |
| model_output | Optional parameter to export RF model trained on train.data.file as JSON and treelite files. This model will *ALWAYS be fitted using all samples of train.data.file* regardless of if LOOCV is carried out for prediction | False |
| bootstrap | Optional boolean to turn on classifier bootstrapping | True |
| split_criterion | Optional string for node-splitting criterion of one of the following: {“gini”, “entropy”} | "gini" |
| max_depth | Optional int for maximum tree depth (>= 1) | 16 |
| class_weight | Optional string for class weight specification of either of: {"balanced," "balanced_subsample"}, also takes None ; (**future implementation:** to handle input of dictionary/list of); Note: "balanced" or "balanced_subsample" are not recommended for warm start if the fitted data differs from the full dataset | None |
| max_features | Optional string for number of features per split of either one of the following: {"auto", "sqrt," "log2"}, (**future implementation:** handle input of float/int) | "auto" |
| max_leaves | Optional int for maximum leaf nodes per tree (>= 1), also takes -1 for unlimited | -1 |
| max_samples | Optional float for ratio of datasets to use per tree (between 0.0 and 1.0, inclusive for both) | 1.0 |
| min_impurity_decrease | Optional float for minimum impurity decrease needed per node split (>= 0.0) | 0.0 |
| min_samples_leaf | Optional int for minimum number of samples required at leaf node (>= 1) | 1 |
| min_samples_split | Optional int for minimum sample number to split node (>= 2) | 2 |
| min_weight_fraction_leaf | Optional float for min weighted fraction of weight sum total to be leaf (between 0.0 and 0.5, inclusive for both) | 0.0 |
| n_estimators | Optional int for number of trees in forest (>= 1) | 100 |
| n_bins | Optional parameter for maximum number of bins used by split algorithm per feature (>=1) | 128 |
| random_state | Optional int for seed of random number generator (nonnegative, caps at 4294967295, 2<sup>32</sup> - 1), also takes None. Note: Setting this to a specific integer, like 0 for example, for a specific dataset, will *NOT* always yield the same results due to RAPIDS.ai limitations | None |
| max_batch_size | Optional int for maximum number of nodes that can be processed in a given batch (>=0) | 4096 |
| debug | Optional boolean for program debugging | False |
| verbose | Optional int (0 = no verbose, 1 = base verbosity) to increase classifier verbosity (non-negative), [more info](https://docs.rapids.ai/api/cuml/stable/api/#verbosity-levels) (for other input values) | 0 |

\*  required

## Input Files (see table above for which to input for a specific mode)

1. Training Data Feature File   
    This is the input file of classifier training feature data which is used to create the random forest model (for the case of test-train prediction, the training dataset can be substituted by a model pickle file input). For cross-validation, this is the only feature data input which also has prediction done against it via LOOCV. The parameter expects a GCT file (.gct) that follows the [GenePattern GCT](https://www.genepattern.org/file-formats-guide#GCT) file standard.
      
2. Training Data Class File   
    This is the input file of classifier training target data which is used to create the random forest model (for the case of test-train prediction, the training dataset can be substituted by a model pickle file input). For cross-validation, this is the only target data input whose values are considered as "true." The parameter expects a CLS file (.cls) that follows the [GenePattern CLS](https://www.genepattern.org/file-formats-guide#CLS) file standard.

3. Model Input File   
    This is the input file of a fitted Scikit RandomForestClassifier model as a compressed pickel (.pkl) file. It can serve as a substitute for the training dataset in the case of test-train prediction, and if both are provided, the model input file takes precedence and is used.
    
4. Testing Data Feature File   
    This is the input file of classifier testing feature data which the random forest model will predict the class values of (only passed in for test-train prediction). The parameter expects a GCT file (.gct) that follows the [GenePattern GCT](https://www.genepattern.org/file-formats-guide#GCT) file standard.
      
5. Testing Data Class File   
    This is the input file of classifier testing target data whose values are considered as "true" (only passed in for test-train prediction). The parameter expects a CLS file (.cls) that follows the [GenePattern CLS](https://www.genepattern.org/file-formats-guide#CLS) file standard.
    
## Output Files

Outputs prediction results (.pred.odf) and feature importance files (.feat.odf) that follows the [GenePattern ODF (Output Description Format)](https://www.genepattern.org/file-formats-guide#ODF) file standard. They contain a specific set of descriptive headers followed by a main data block comparing the random forest classification's predictions on the entire feature dataset against the true values with confidence scores (using [Scikit RandomForestClassifier _predict_proba](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba)) for the pred.odf file and a main data block of all the features and their importance scores (sum of all is 1) in the case of the feat.odf file.

## Test-Train Example Data

ALL_AML Dataset Inputs (**<ins>Without</ins>** Model Input):
[all_aml_train.gct](/data/all_aml_train.gct), [all_aml_train.cls](/data/all_aml_train.cls), [all_aml_test.gct](/data/all_aml_train.gct), and [all_aml_test.cls](/data/all_aml_train.cls)
ALL_AML Example Outputs:
[all_aml_tt_dataset.pred.odf](/data/example_output/all_aml_tt_dataset.pred.odf), [all_aml_tt_dataset.feat.odf](/data/example_output/all_aml_tt_dataset.feat.odf), and [all_aml_train.pkl](/data/example_output/all_aml_train.pkl)

ALL_AML Dataset Inputs (**<ins>With</ins>** Model Input):
[all_aml_train.pkl](/data/example_output/all_aml_train.pkl), [all_aml_test.gct](/data/all_aml_train.gct), and [all_aml_test.cls](/data/all_aml_train.cls)  
ALL_AML Example Outputs:
[all_aml_tt_model.pred.odf](/data/example_output/all_aml_tt_model.pred.odf), [all_aml_tt_with_model.feat.odf](/data/example_output/all_aml_tt_model.feat.odf) (only if model.output is True and the training dataset is provided, which it is not in this case, will there be an output pickle file of training data)


## Cross-Validation Example Data

ALL_AML Dataset Inputs:
[all_aml_train.gct](/data/all_aml_train.gct) and [all_aml_train.cls](/data/all_aml_train.cls)
ALL_AML Example Outputs:
[all_aml_xval.pred.odf](/data/example_output/all_aml_xval.pred.odf) and [all_aml_train.pkl](/data/example_output/all_aml_train.pkl)

BRCA_HUGO Dataset Inputs:
[DP_4_BRCA_HUGO_symbols.preprocessed.gct](/data/DP_4_BRCA_HUGO_symbols.preprocessed.gct) and [Pred_2_BRCA_HUGO_symbols.preprocessed.cls](/data/Pred_2_BRCA_HUGO_symbols.preprocessed.cls)  
BRCA_HUGO Example Outputs:
[BRCA_xval.pred.odf](/data/example_output/BRCA_xval.pred.odf) and [BRCA.pkl](/data/example_output/BRCA.pkl)

Iris Dataset Inputs:
[iris.gct](/data/iris.gct) and [iris.cls](/data/iris.cls)  
Iris Example Outputs:
[iris_xval.pred.odf](/data/example_output/iris_xval.pred.odf) and [iris.pkl](/data/example_output/iris.pkl)


## Requirements

Requires the following NMF-GPU Singularity container found [here](https://github.com/genepattern/nmf-gpu/blob/master/docker/Singularity.def) (created from Dockerfile).

## Miscellaneous

Future development ideas:
* Handling the following miscellaneous input arguments: class_weight input of dictionary/list of; max_features input of int/float
* Implementation of feature importance results file


## License

`RandomForestGPU` is distributed under a modified BSD license available [here](/LICENSE.txt)

## Version Comments

| Version | Release Date | Description                                 |
----------|--------------|---------------------------------------------|
| 1.0.0 | Sept 15, 2023 | Initial version for team use. |
