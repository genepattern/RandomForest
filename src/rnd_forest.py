#!/usr/bin/env python3

# Importing file processing functions
from rnd_forest_functions import *
# Importing class to differentiate feature & target logic
from Marker import *

# Importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse as ap

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     rnd_forest.py
    Project:       RandomForest (Non-GPU)
    Description:   Non-GPU RandomForest main python script to: 
                    - Process optional classifier arguments
                    - Validate files for feature (.gct) & target (.cls) data
                    - Call functions to process files into DataFrames
                    - Perform Random Forest classification.
                   Outputs accuracy as well as a results (.pred.odf) file.
                   Designed to allow for further file type implementation.
                   Created as GenePattern module.
                   
    References:    scholarworks.utep.edu/cs_techrep/1209/
                   datacamp.com/tutorial/random-forests-classifier-python
                   tiny.cc/7cl2vz
"""

# Helper method to account for str type or None input
def none_or_str(value):
    if value == 'None':
        return None
    return value

# Helper method to account for int type or None input
def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

# Helper method to account for float type or None input
def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

# Helper method to account for command line boolean input
def str_bool(value):
    # Checking for string input ("True" or "False")
    if (type(value) == str):
        # Returning boolean counterpart
        return eval(value)
    else:
        return value
        

# Adding arguments to script for classifier feature & target file input,
# .pred.odf file output, scikit RandomForest classifier parameters, & debugging
parser = ap.ArgumentParser(description='Scikit Random Forest Classifier')

# Adding file input arguments (required)
# Feature file input (.gct):
parser.add_argument("-f", "--feature", help="classifier feature data filename"
                    + " Valid file format(s): .gct", required=True)
# Target file input (.cls):
parser.add_argument("-t", "--target", help="classifier target data filename"
                    + " Valid file format(s): .cls", required=True)

# Target file input (.pred.odf) (optional as default value exists):
parser.add_argument("-p", "--pred_odf", help="prediction output filename",
                    nargs="?", const=1)


# Random Forest Classifier arguments (optional) as default values exist for all:
# Test/Training set split argument, 30% for test is default (70% for training):
# Range is floats from 0.0 to 1.0, exclusive of both
parser.add_argument("--test_size", 
                    help="ratio for test data split, rest is training data",
                    nargs="?", const=1, default=0.3, type=float)

# Either True or False
parser.add_argument("--bootstrap", help="boolean for bootstrapping",
                    nargs="?", const=1, default=True, type=str_bool)

# Range is floats greater than or equal to 0.0
parser.add_argument("--ccp_alpha", 
                    help="complexity parameter of min cost-complexity pruning",
                    nargs="?", const=1, default=0.0, type=float)

# TODO, is dict/lis of, but generally,is "balanced" or "balanced_subsample"
# Can also be None (default value)
parser.add_argument("--class_weight", help="class weight specification",
                    nargs="?", const=1, default=None, type=none_or_str)

# Values are "gini," "entropy," or "log_loss"
parser.add_argument("--criterion", help="criterion of node splitting",
                    nargs="?", const=1, default="gini", type=str)

# Range is integers greater than or equal to 1
# Can also be None (default value)
parser.add_argument("--max_depth", help="maximum tree depth",
                    nargs="?", const=1, default=None, type=none_or_int)

# TODO, can be "sqrt", "log2", "auto" ("auto" removed in 1.3), a float, or an int
parser.add_argument("--max_features", 
                    help="number (ratio in cuML) of features per split",
                    nargs="?", const=1, default="sqrt", type=str)

# Range is all integers betwen greater than or equal to 2
# Also takes None (default value)
parser.add_argument("--max_leaf_nodes", help="maximum leaf nodes per tree",
                    nargs="?", const=1, default=None, type=none_or_int)

# Range is all floats betwen 0.0 and 1.0, inclusive of both
# Also takes None (default value)
parser.add_argument("--max_samples", 
                    help="number (ratio for cuML) of datasets to use per tree",
                    nargs="?", const=1, default=None, type=none_or_float)

# Range is all floats greater than or equal to 0.0
parser.add_argument("--min_impurity_decrease", 
                    help="minimum impurity decrease needed per node split",
                    nargs="?", const=1, default=0.0, type=float)

# Range is all ints greater than or equal to 1
# Using integer implementation [1, inf) and NOT float implementation (0.0, 1.0)
parser.add_argument("--min_samples_leaf", 
                    help="minimum number of samples required at leaf node",
                    nargs="?", const=1, default=1, type=int)

# Range is all ints greater than or equal to 2
# Using integer implementation [2, inf) and NOT float implementation (0.0, 1.0)
parser.add_argument("--min_samples_split", 
                    help="minimum sample number to split node",
                    nargs="?", const=1, default=2, type=int)

# Range is all floats between 0.0 and 0.5, inclusive of both
parser.add_argument("--min_weight_fraction_leaf", 
                help="min weighted fraction of weight sum total to be leaf",
                nargs="?", const=1, default=0.0, type=float)

# Range is all integers greater than or equal to 1
parser.add_argument("--n_estimators", 
                    help="number of trees in forest",
                    nargs="?", const=1, default=100, type=int)

# Note, n_jobs=0 has no meaning in Random Forest Classifier, -1 for all CPUs
# Range is all nonzero integers, can also be None (default value)
parser.add_argument("--n_jobs", 
                    help="number of parallel streams for building the forest",
                    nargs="?", const=1, default=None, type=none_or_int)

# Either True or False
parser.add_argument("--oob_score",
                    help="if out-of-bag samples used for generalization score",
                    nargs="?", const=1, default=False, type=str_bool)

# Range is all ints between 0 and max seed (2^32 - 1 = 4294967295),
# inclusive of both, can also be None (default value)
max_seed = (2**32) - 1
parser.add_argument("--random_state", 
                    help="seed for random number generator",
                    nargs="?", const=1, default=None, type=none_or_int)

# Either True or False
parser.add_argument("--warm_start", 
                    help="whether to start new forest or add to past solution",
                    nargs="?", const=1, default=False, type=str_bool)
                    
# 0 for no verbosity, 1 for basic verbosity, values greater for more verbosity
# Range is integers greater than or equal to 0
parser.add_argument("-v", "--verbose", help="classifier verbosity flag",
                    nargs="?", const=1, default=0, type=int)

# Program debug argument, either True or False, False by default
parser.add_argument("-d", "--debug", help="program debug messages",
                    nargs="?", const=1, default=False, type=str_bool)

# Parsing arguments for future calls within script to utilize
args = parser.parse_args()


# Verifying debug status
if (args.debug):
    print("Debugging on.")
    print()


# Checking for feature and target data file validity by calling file_valid
# Obtained output corresponds to file extension if valid, None if invalid
feature_ext = file_valid(args.feature, Marker.FEAT)
target_ext = file_valid(args.target, Marker.TAR)

# Only carrying out Random Forest Classification if both files are valid
if ((feature_ext != None) and (target_ext != None)):

    # Processing the valid files into dataframes with parent function "process"
    feature_df = process(args.feature, feature_ext, Marker.FEAT)
    target_df = process(args.target, target_ext, Marker.TAR)

    # Creating instance of Random Forest Classifier with arguments parsed
    clf = RandomForestClassifier(
        bootstrap=args.bootstrap, ccp_alpha=args.ccp_alpha, 
        class_weight=args.class_weight, criterion=args.criterion,
        max_depth=args.max_depth, max_features=args.max_features,
        max_leaf_nodes=args.max_leaf_nodes, max_samples=args.max_samples,
        min_impurity_decrease=args.min_impurity_decrease,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
        n_estimators=args.n_estimators, n_jobs=args.n_jobs, 
        oob_score=args.oob_score, random_state=args.random_state, 
        verbose=args.verbose, warm_start=args.warm_start)
    
    # Printing list of passed-in values if verbosity is on (value greater than 0)
    print("Test/Training Split is: ", 
        (args.test_size) * 100, "/", (1-args.test_size) * 100)
    print()
    print("Classifier parameter values: ")
    print(clf.get_params(True))
    print()

    # Setting values to test and training feature and target dataframes
    # For 30/70 Test/Training split default, see first source
    X_train, X_test, y_train, y_test = train_test_split(feature_df.T,
                                    target_df.T, test_size=args.test_size)

    # Training the model with training sets of X and y
    # Raveling y_train's values for data classification format, see last reference
    clf.fit(X_train, y_train.values.ravel())

    # Predicting using entire feature data
    pred=clf.predict(feature_df.T)

    # Initializing variable for true target values
    true = target_df.iloc[0].values

    # Classifier accuracy check
    accuracy = accuracy_score(true, pred) * 100
    print(f"Accuracy score: " + "{0:.2f}%".format(accuracy))
    print()


    if (args.debug):

        print("Feature DataFrame")
        print (feature_df)
        print()

        print("Target DataFrame")
        print (target_df)
        print()

        print("Predicted target values, done on all feature data:")
        print (pred)
        print()

        print("X_train:")
        print (X_train)
        print()

        print("X_test:")
        print (X_test)
        print()

        print("y_train:")
        print (y_train)
        print()

        print("y_test:")
        print (y_test)
        print()

    # If no value was provided for pred_odf filename, uses name of feature file
    if (args.pred_odf == None):
        pred_odf = pred_filename(args.feature)
    # Else, uses passed-in name
    else:
        pred_odf = args.pred_odf   

    # Creating pred.odf dataframe
    df = pd.DataFrame(columns=range(true.size))

    # Initializing counter for mismatches
    counter = 0

    # Initializing array for target name values (e.g, ["all", "aml"])
    tar = tar_array(args.target, target_ext)

    # Iterating through each sample
    for i in range(0,true.size):

        # Creating boolean for mismatch check
        check = true[i] == pred[i]
        
        if (check == True):
            value = "TRUE"
        else:
            value = "FALSE"
            counter+=1

        # Assigning true's and preds's values to the respective sample values
        # and evaluating differences. Using tar array to specify target names
        df[i] = [list(feature_df.columns)[i], tar[true[i]],
                tar[pred[i]], 1, value]

    # Creating dictionary for pred_odf file header
    header_dict = {
        "HeaderLines" : "",
        "COLUMN_NAMES" : "Samples\t" + "True Class\t" + "Predicted Class\t" + "Confidence\t" + "Correct?",
        "COLUMN_TYPES" : "String \t" + "String\t" + "String\t" + "float\t" + "boolean",
        "Model" : "Prediction Results",
        "PredictorModel" : "Random Forest Classifier",
        "NumFeatures" : 0,
        "NumCorrect" : (true.size - counter),
        "NumErrors" : counter,
        "DataLines" : true.size
    }

    # Passing transposed odf dataframe and header_dict into GP write_odf()
    write_odf(df.T, pred_odf, header_dict)

# Otherwise, printing error message to notify user (in dev case CLI-usage)
else:
    print("Error in file input, please check above for details.")

