#!/usr/bin/env python3

# Importing file processing functions
from rnd_forest_functions import *
# Importing class to differentiate feature & target logic
from Marker import *

# Importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import argparse as ap

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     rnd_forest.py
    Project:       RandomForest (Non-GPU)
    Description:   Non-GPU RandomForest main python script to: 
                    - Process optional classifier arguments
                    - Validate feature (.gct) & target (.cls) data file inputs
                    - Call functions to process files into DataFrames
                    - Perform Random Forest classification using either:
                      leave-one-out cross-validation (given 2 files) or 
                      test-train prediction (given 4 files)
                    - Predict feature dataset and compare to "true" target file
                   Outputs accuracy (to stdout) and a results (.pred.odf) file.
                   Designed to allow for further file type implementation.
                   Created for GenePattern module usage.
                   
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


# Test-train classification arguments (optional as can opt for LOOCV):
# Feature file input (.gct):
parser.add_argument("--test_feat", help="classifier test feature data filename"
                    + " Valid file format(s): .gct")
# Target file input (.cls):
parser.add_argument("--test_tar", help="classifier test target data filename"
                    + " Valid file format(s): .cls")


# Random Forest Classifier arguments (optional) as defaults exist for all:
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

# TODO, can be "sqrt", "log2", "auto" ("auto" removed in 1.3), a float, or int
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
    print("Debugging on.\n")


# Checking for feature and target data file validity by calling file_valid
# Obtained output corresponds to file extension if valid, None if invalid
# Train input (required)
feature_ext = file_valid(args.feature, Marker.FEAT)
target_ext = file_valid(args.target, Marker.TAR)
# Test input (optional)
test_feat_ext = file_valid(args.test_feat, Marker.FEAT)
test_tar_ext = file_valid(args.test_tar, Marker.TAR)

# Only carrying out Random Forest Classification if both files are valid
if ((feature_ext != None) and (target_ext != None)):

    # Processing the valid files into dataframes with parent function "process"
    feature_df = process(args.feature, feature_ext, None, Marker.FEAT)
    target_df = process(args.target, target_ext, None, Marker.TAR)

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
        oob_score=args.oob_score, random_state=args.random_state)
        
    if (args.debug):
        print(clf.get_params(deep=True), "\n")

    # Creating array for holding target prediction values
    pred_arr = []

    # Case of no test dataset provided, so doing LOOCV
    if ((test_feat_ext == None) and (test_tar_ext == None)):
        
        # Instantiating variable to hold value of column names
        # to use for prediction results
        cols = feature_df.columns

        # Creating instance of Leave-One-Out Cross Validation
        loo = LeaveOneOut()
        
        if (args.debug):
            print("Number of splitting iterations in LOOCV:", 
                    loo.get_n_splits(feature_df.T), "\n")
            print("Feature DataFrame: \n", feature_df, "\n")
            print("Target DataFrame: \n", target_df, "\n\n")

        # Iterating through each sample to do RF Classification by LOOCV
        for i, (train_index, test_index) in enumerate(loo.split(feature_df.T)):

            # Obtaining column and row names
            col = feature_df.columns[i]
            row = target_df.columns[i]

            # Doing LOOCV by creating training data without left-out sample
            X_train = (feature_df.drop(col, axis=1)).T
            y_train = (target_df.drop(row, axis=1)).T

            # Training the model with training sets of X and y
            # Raveling y_train for data classification format, see last source
            clf.fit(X_train, y_train.values.ravel())

            # Initialzing iteration's X_test value
            # Reshaping necessary as array always 1D (single sample's data)
            X_test = (feature_df.loc[:,col].T).values.reshape(1, -1)

            # Predicting target value of left-out sample feature data
            pred = clf.predict(X_test)

            # Using [0] for X_test and pred for formatting
            if (args.debug):
                print("Sample:", i, "\n")
                print ("Feature training set with debug sample removed:\n",
                    X_train, "\n")
                print("Target training set with debug sample removed:\n",
                    y_train, "\n")
                print ("LOOCV run feature testing data (debug sample):\n",
                    X_test[0], "\n")
                print("LOOCV run target pred. (debug sample prediction):\n",
                    pred[0], "\n\n")

            # Appending prediction to list (pred is array, hence pred[0])
            pred_arr.append(pred[0])

        # Initializing variable for true target values (train in this case)
        true = target_df.iloc[0].values


    # Case of test-train prediction (4 files given, 2 test and 2 train)
    else:

        # Processing test files into dataframes with parent function "process"
        test_feat_df = process(args.test_feat, feature_ext, args.feature, 
                               Marker.FEAT)
        test_tar_df = process(args.test_tar, target_ext, None, Marker.TAR)

        # Instantiating variable to hold value of column names
        #  to use for prediction results
        cols = test_feat_df.columns

        # Assigning variables for training feature and target data
        X_train = feature_df.T
        y_train = target_df.T

        # Assigning variable for testing feature and target data
        X_test = test_feat_df.T
        y_test = test_tar_df.T

        if (args.debug):
            print("Training Feature DataFrame: \n", feature_df, "\n")
            print("Training Target DataFrame: \n", target_df, "\n\n")
        if (args.debug):
            print("Testing Feature DataFrame: \n", test_feat_df, "\n")
            print("Testing Target DataFrame: \n", test_tar_df, "\n\n")
    
        # Training the model with training sets of X and y
        # Raveling y_train for data classification format, see last source
        clf.fit(X_train, y_train.values.ravel())

        # Predicting using test features
        y_pred=clf.predict(X_test)

        # Initializing variable for true target values (test in this case)
        true = test_tar_df.iloc[0].values

        pred_arr = y_pred


    # If no value was provided for pred_odf filename, 
    # uses name of feature target (or test, if provided) file:
    if ((args.pred_odf == None) and (args.test_feat != None)):
        pred_odf = pred_filename(args.test_feat)
    else:
        pred_odf = args.pred_odf

    if (args.debug):
        print("True target values:\n", *true, "\n", sep=" ")
        print("Predicted target values:\n", *pred_arr, "\n", sep=" ")
    
    if (args.debug):
        # Classifier accuracy check
        accuracy = accuracy_score(true, pred_arr) * 100
        print(f"Accuracy score: " + "{0:.2f}%".format(accuracy), "\n") 

    # Creating pred.odf dataframe
    df = pd.DataFrame(columns=range(true.size))

    # Initializing counter for mismatches
    counter = 0

    # Initializing array for target name values (e.g, ["all", "aml"])
    tar = tar_array(args.target, target_ext)

    if (args.debug):
        print("Target names array:", tar, "\n")


    # Iterating through each sample
    for i in range(0,true.size):

        # Creating boolean for mismatch check
        check = true[i] == pred_arr[i]
        
        if (check == True):
            value = "TRUE"
        else:
            value = "FALSE"
            counter+=1

        # Assigning true's and preds's values to the respective sample values
        # and evaluating differences. Using tar array to specify target names
        df[i] = [i+1, list(cols)[i], tar[true[i]],
                tar[pred_arr[i]], 1, value]

    # Creating dictionary for pred_odf file header
    header_dict = {
        "HeaderLines" : "",
        "COLUMN_NAMES" : "Samples\t" + "True Class\t" + "Predicted Class\t" 
            + "Confidence\t" + "Correct?",
        "COLUMN_TYPES" : "String \t" + "String\t" + "String\t" + "float\t" 
            + "boolean",
        "Model" : "Prediction Results",
        "PredictorModel" : "Random Forest Classifier",
        "NumFeatures" : 0,
        "NumCorrect" : (true.size - counter),
        "NumErrors" : counter,
        "DataLines" : true.size
    }

    if (args.debug):
        print("Output filename: " + pred_odf)

    # Passing transposed odf dataframe and header_dict into GP write_odf()
    write_odf(df.T, pred_odf, header_dict)

# Otherwise, printing error message to notify user (in dev case CLI-usage)
else:
    print("Error in file input, please check above for details.")
