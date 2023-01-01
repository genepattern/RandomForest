# RandomForest (Non-GPU)

**Description**: The following is a GenePattern module written in Python 3. It performs random forest classification on given feature and target data file inputs using Scikit-learn's RandomForestClassifier. Also includes several optional parameters for specifying the ranodm forst algorithm classification process. This module/repo serves as a foundation for implementing the cuML-based GPU Random Forest Classifier.

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Summary

This repository is a GenePattern module written in [Python 3](https://www.python.org/download/releases/3.0/).

It takes in two files, one for classifier feature data (.gct), and one for classifier target data (.cls). Then, it processes them into DataFrames and performs classification on them using Scikit-learn's RandomForestClassifier, generating an accuracy score. Created for both command-line and module usage through optional arguments for classifier parameters. Designed for smooth implementation of other file type inputs (.txt input, etc...) and future features (including optional prediction of other user-provided feature data).


## Source Links
* [The GenePattern RandomForest source repository](https://github.com/omarhalawa3301/randomforest)
* RandomForest uses the [genepattern/notebook-python39:22.04](https://hub.docker.com/layers/genepattern/notebook-python39/22.04/images/sha256-1182e33d0a4d944e676003b2d4a410ec3a197db13847292cedca441a0541513d?context=explore)

## Usage
python rnd_forest.py **-f** &lt;feature datafile&gt; **-t** &lt;target datafile> -d --bootstrap True --ccp_alpha 0.0 --class_weight None --criterion gini --max_depth None --max_features sqrt --max_leaf_nodes None --max_samples None --min_impurity_decrease 0.0 --min_samples_leaf 1 --min_samples_split 2 --min_weight_fraction_leaf 0.0 --n_estimators 100 --n_jobs None --oob_score False --random_state None --verbose 0 --warm_start False

## Parameters

| Name | Description | Default Value |
---------|--------------|----------------
| filename * |  The input file to be read in .gct format | No default value |
| verbose | Optional parameter to increase output verbosity | False |

\*  required

## Input Files

1. filename  
    This is the input file which will be read in by the python script and ultimately will be processed through log normalization of positive values. The parameter expects a GCT file (.gct extension).
    
## Output Files

1. result.gct\
    The log-normalized version of the input GCT file's data. Non-positive values all become 0.
2. stdout.txt\
    This is standard output from the Python script. Sometimes helpful for debugging.

## Example Data

Input:  
[all_aml_train.gct](https://github.com/omarhalawa3301/log_normalize/blob/main/data/all_aml_train.gct)

Output:  
[result.gct](https://github.com/omarhalawa3301/log_normalize/blob/main/data/result.gct)


## Requirements

Requires the [genepattern/notebook-python39:latest Docker image](https://hub.docker.com/layers/genepattern/notebook-python39/21.08/images/sha256-12b175ff4472cfecef354ddea1d7811f2cbf0ae9f114ede11d789b74e08bbc03?context=explore).

## License

`LogTransform` is distributed under a modified BSD license available [here](https://github.com/omarhalawa3301/log_normalize/blob/main/LICENSE.txt)

## Version Comments

| Version | Release Date | Description                                 |
----------|--------------|---------------------------------------------|
| 1.0.0 | Oct 4, 2022 | Initial version for team use. |
