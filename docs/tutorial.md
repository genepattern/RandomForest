# RandomForest (Non-GPU)

**Description**: The following is a GenePattern module written in Python 3. It performs random forest classification on given feature and target data file inputs using Scikit-learn's RandomForestClassifier. Also includes several optional parameters for specifying the random forest algorithm classification process. This module/repo serves as a foundation for implementing the cuML-based GPU Random Forest Classifier.

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Summary

This repository is a GenePattern module written in [Python 3](https://www.python.org/download/releases/3.0/).

It takes in two files, one for classifier feature data (.gct), and one for classifier target data (.cls). Then, it processes them into DataFrames and performs classification on them using Scikit-learn's RandomForestClassifier, generating an accuracy score. Created for both command-line and module usage through optional arguments for classifier parameters. Designed for smooth implementation of other file type inputs (.txt input, etc...) and future features (including optional prediction of other user-provided feature data).


## Source Links
* [The GenePattern RandomForest source repository](https://github.com/omarhalawa3301/randomforest)
* RandomForest uses the [genepattern/notebook-python39:22.04](https://hub.docker.com/layers/genepattern/notebook-python39/22.04/images/sha256-1182e33d0a4d944e676003b2d4a410ec3a197db13847292cedca441a0541513d?context=explore)

## Usage
* Basic command-line usage: 
 
        python rnd_forest.py -f <feature datafile> -t <target datafile>

* Full command-line usage (optional arguments added): 

        python rnd_forest.py -f <feature datafile> -t <target datafile> (-d) (--test_size [float]) (--bootstrap [bool]) 
        (--ccp_alpha [float]) (--class_weight [str or None]) (--criterion [str]) (--max_depth [int or None])
        (--max_features [str]) (--max_leaf_nodes [int or None]) (--max_samples [float or None]) (--min_impurity_decrease [float]) 
        (--min_samples_leaf [int]) (--min_samples_split [int]) (--min_weight_fraction_leaf [float]) (--n_estimators [int])
        (--n_jobs [int or None]) (--oob_score [bool]) (--random_state [int or None]) (--verbose [int]) (--warm_start [bool])
        
* Example with all arguments set to defaults

        python rnd_forest.py -f iris.gct -t iris.cls -d --test_size 0.3 --bootstrap True --ccp_alpha 0.0 --class_weight None
       --criterion gini --max_depth None --max_features sqrt --max_leaf_nodes None --max_samples None --min_impurity_decrease 0.0
       --min_samples_leaf 1 --min_samples_split 2 --min_weight_fraction_leaf 0.0 --n_estimators 100 --n_jobs None --oob_score False
       --random_state None --verbose 0 --warm_start False


## Parameters

| Name | Description | Default Value |
---------|--------------|----------------
| feature (-f) * |  Classifier feature data filename to be read from user (.gct, more format support to come) | No default value |
| target (-t) * |  Classifier target data filename to be read from user (.cls, more format support to come) | No default value |
| debug (-d) | Optional flag to turn on program debugging | False |
| test_size | Optional float for ratio of total data split for testing (for test/training data split, rest for training), (between 0.0 and 1.0, exclusive for both) | 0.3 |
| bootstrap | Optional boolean to turn on classifier bootstrapping | True |
| ccp_alpha | Optional float for complexity parameter of min cost-complexity pruning (>= 0.0) | 0.0 |
| class_weight | Optional string for class weight specification of either of: {"balanced," "balanced_subsample"}, also takes None ("None" in CLI); (**future implementation:** handle input of dictionary/list of) | None |
| criterion | Optional string for node-splitting criterion of one of the following: {“gini”, “entropy”, “log_loss”} | "gini" |
| max_depth | Optional int for maximum tree depth (>= 1), also takes None ("None" in CLI) | None |
| max_features | Optional string for number of features per split of either one of the following: {"sqrt," "log2"} ("auto" to be removed in Scikit 1.3), (**future implementation:** handle input of float/int) | "sqrt" |
| max_leaf_nodes | Optional int for maximum leaf nodes per tree (>= 2), also takes None ("None" in CLI) | None |
| max_samples | Optional float for number of datasets to use per tree (between 0.0 and 1.0, inclusive for both), also takes None ("None" in CLI) | None |
| min_impurity_decrease | Optional float for minimum impurity decrease needed per node split (>= 0.0) | 0.0 |
| min_samples_leaf | Optional int for minimum number of samples required at leaf node (>= 1) | 1 |
| min_samples_split | Optional int for minimum sample number to split node (>= 2) | 2 |
| min_weight_fraction_leaf | Optional float for min weighted fraction of weight sum total to be leaf (between 0.0 and 0.5, inclusive for both) | 0.0 |
| n_estimators | Optional int for number of trees in forest (>= 1) | 100 |
| n_jobs | Optional int for number of parallel streams for building the forest (nonzero), also takes None ("None" in CLI), [more info](https://scikit-learn.org/stable/glossary.html#term-n_jobs) (-1 for all CPUs) | None |
| oob_score | Optional boolean for if out-of-bag samples used for generalization score | False |
| random_state | Optional int for seed of random number generator (nonnegative, caps at 4294967295, 2<sup>32</sup> - 1), also takes None ("None" in CLI) | None |
| verbose | Optional int (0 = no verbose, 1 = base verbosity) to increase classifier verbosity (non-negative), [more info](https://scikit-learn.org/stable/glossary.html#term-verbose) (for other input values) | 0 |
| warm_start | Optional boolean for whether to start new forest or add to past solution | False |

\*  required

## Input Files

1. feature datafile  
    This is the input file of classifier feature data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a GCT file (.gct), but future support for other feature data formats will be implemented.  
      
2. target datafile  
    This is the input file of classifier target data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a CLS file (.cls), but future support for other feature data formats will be implemented.  

    
## Output Files

No output files in current implementation (future versions to include optional file output of prediction done on user-provided feature data)

## Example Data

Iris Dataset Inputs:
[iris.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.gct) and [iris.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.cls)  
Iris Example Output:
[example_output_iris.txt](https://github.com/omarhalawa3301/randomforest/blob/main/data/example_output/example_output_iris.txt)


ALL_AML Dataset Inputs:
[all_aml_train.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.gct) and [all_aml_train.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.cls)  
ALL_AML Example Output:
[example_output_all_aml.txt](https://github.com/omarhalawa3301/randomforest/blob/main/data/example_output/example_output_all_aml.txt)

## Requirements

Requires the [genepattern/notebook-python39:22.04 Docker image](https://hub.docker.com/layers/genepattern/notebook-python39/22.04/images/sha256-1182e33d0a4d944e676003b2d4a410ec3a197db13847292cedca441a0541513d?context=explore).

## Miscellaneous

Future development ideas: Prediction on other user-input feature data

## License

`RandomForest` is distributed under a modified BSD license available [here](https://github.com/omarhalawa3301/randomforest/blob/main/LICENSE.txt)

## Version Comments

| Version | Release Date | Description                                 |
----------|--------------|---------------------------------------------|
| 1.0.0 | Jan 2, 2023 | Initial version for team use. |
