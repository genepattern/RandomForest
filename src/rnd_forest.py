#!/usr/bin/env python3

# Importing file processing functions
from rnd_forest_functions import *
# Importing class to differentiate feature & target logic & specify module mode
from Marker import *

# Importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import argparse as ap
import joblib
import re

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
                      test-train prediction (given 4 files or 2 files & model)
                    - Predict feature dataset and compare to "true" target file
                   Outputs results (.pred.odf) & feature importance (.feat.odf)
                   (feature importance only for test-train prediction)
                   Designed to allow for further file type implementation.
                   Created for GenePattern module usage.
                   
    References:    scholarworks.utep.edu/cs_techrep/1209/
                   datacamp.com/tutorial/random-forests-classifier-python
                   tiny.cc/7cl2vz
"""

###############################################################################

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
    
###############################################################################
        
# Adding arguments to script for classifier feature & target file input,
# .pred.odf & .feat.odf outputs, scikit RFC parameters, & debugging
parser = ap.ArgumentParser(description='Scikit Random Forest Classifier')

# Adding file input arguments (combinations of these dictate mode)
# Feature file input (.gct):
parser.add_argument("-f", "--feature", help="classifier feature data file"
                    + " Valid file format(s): .gct")

# Target file input (.cls):
parser.add_argument("-t", "--target", help="classifier target data file"
                    + " Valid file format(s): .cls")

# Feature file input (.gct):
parser.add_argument("--test_feat", help="classifier test feature data file"
                    + " Valid file format(s): .gct")

# Target file input (.cls):
parser.add_argument("--test_tar", help="classifier test target data file"
                    + " Valid file format(s): .cls")

# Assigning results file's name (.pred.odf):
parser.add_argument("--pred_odf", help="prediction results filename")

# Assigning results file's name (.feat.odf):
parser.add_argument("--feat_odf", help="feature importances filename" +
                    "this is only outputted for the test-train prediction mode")

# Output model of training data (.pkl file output):
parser.add_argument("--model_output", help="output model of training data",
                    nargs="?", const=1, default=False, type=str_bool)

# Output model of training data (.pkl file output):
parser.add_argument("--model_output_filename", help="name of output model")

# Input model (.pkl file output)
parser.add_argument("--model_input", help="input model for training data")

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

# TODO, can be "sqrt", "log2", "None" ("auto" removed in 1.3), a float, or int
parser.add_argument("--max_features", 
                    help="number of features to consider for best split",
                    nargs="?", const=1, default="sqrt", type=none_or_str)

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

###############################################################################

# Checking for feature and target data file validity by calling file_valid
# Obtained output corresponds to file extension if valid, None if invalid
# Train input (required)
train_feat_ext = file_valid(args.feature, Marker.FEAT)
train_tar_ext = file_valid(args.target, Marker.TAR)
# Test & model input (optional)
test_feat_ext = file_valid(args.test_feat, Marker.FEAT)
test_tar_ext = file_valid(args.test_tar, Marker.TAR)
model_in_ext = file_valid(args.model_input, Marker.MODEL)

# Establishing what mode to run
if all(item != None for item in [train_feat_ext,  train_tar_ext, 
                                 test_feat_ext, test_tar_ext]):
    mode = Marker.TT
elif all(item != None for item in [model_in_ext, test_feat_ext, test_tar_ext]):
    mode = Marker.TT
elif all(item != None for item in [train_feat_ext,  train_tar_ext]): 
    mode = Marker.LOOCV
else:
    mode = Marker.FAIL

model_exists = model_in_ext != None

if (args.debug):
    print(mode)

###############################################################################

# Input validity check
if (mode != Marker.FAIL):

    # If using a training dataset and not a model input
    if (not model_exists or mode == Marker.LOOCV):
        
        # Processing valid files into dataframes with parent function "process"
        train_feat_df = process(args.feature, train_feat_ext, None, Marker.FEAT)
        train_tar_df = process(args.target, train_tar_ext, None, Marker.TAR)

        # Obtaining feature names and descriptions
        row_names = train_feat_df["Name"].tolist()
        row_desc = train_feat_df["Description"].tolist()
        # Cleaving them off after
        train_feat_df = train_feat_df.set_index("Name")
        train_feat_df.pop("Description")

        # Creating instance of Random Forest Classifier with arguments parsed
        clf = RandomForestClassifier(
            bootstrap=args.bootstrap, ccp_alpha=args.ccp_alpha, 
            class_weight=args.class_weight, criterion=args.criterion,
            max_depth=args.max_depth, max_features=args.max_features,
            max_leaf_nodes=args.max_leaf_nodes, max_samples=args.max_samples,
            min_impurity_decrease=args.min_impurity_decrease,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split, verbose=args.verbose,
            min_weight_fraction_leaf=args.min_weight_fraction_leaf,
            n_estimators=args.n_estimators, n_jobs=args.n_jobs,
            oob_score=args.oob_score, random_state=args.random_state)
            
        if (args.debug):
            print(clf.get_params(deep=True), "\n")

    # Creating array for holding target prediction values
    pred_arr = []
    proba_arr = []

    ###########################################################################

    # Case of no test dataset provided, so doing LOOCV
    if (mode == Marker.LOOCV):
        
        # Instantiating variable to hold value of column names
        # to use for prediction results
        cols = train_feat_df.columns

        # Creating instance of Leave-One-Out Cross Validation
        loo = LeaveOneOut()
        
        if (args.debug):
            print("Number of splitting iterations in LOOCV:", 
                    loo.get_n_splits(train_feat_df.T), "\n")
            print("Feature DataFrame: \n", train_feat_df, "\n")
            print("Target DataFrame: \n", train_tar_df, "\n\n")

        # Iterating through each sample to do RF Classification by LOOCV
        for i, (train_index, test_index) in enumerate(loo.split(train_feat_df.T)):

            # Obtaining column and row names
            col = train_feat_df.columns[i]
            row = train_tar_df.columns[i]
            
            # Doing LOOCV by creating training data without left-out sample
            X_train = (train_feat_df.drop(col, axis=1)).T
            y_train = (train_tar_df.drop(row, axis=1)).T

            # Training the model with training sets of X and y
            # Raveling y_train for data classification format, see last source
            clf.fit(X_train, y_train.values.ravel())

            # Initialzing iteration's X_test value
            # Reshaping necessary as array always 1D (single sample's data)
            X_test = (train_feat_df.loc[:,col].T).values.reshape(1, -1)

            # Predicting target value of left-out sample feature data
            pred = clf.predict(X_test)
            proba = clf.predict_proba(X_test)

            # Using [0] for X_test and pred for formatting
            if (args.debug):
                print("Sample:", col, "\n")
                print ("Feature training set with debug sample removed:\n",
                    X_train.T, "\n")
                print("Target training set with debug sample removed:\n",
                    y_train.T, "\n")
                print ("LOOCV run feature testing data (debug sample):\n",
                    X_test[0].T, "\n")
                print("LOOCV run target pred. (debug sample prediction):\n",
                    pred[0].T, "\n\n")

            # Appending prediction to list (pred is array, hence pred[0])
            pred_arr.append(pred[0])
            proba_arr.append([max(sublist) for sublist in proba][0])

        # Initializing variable for true target values (train in this case)
        true = train_tar_df.iloc[0].values

        # Fitting model to entire feature dataset if a model is to be output
        # (else would have to output one of the LOOCV models)
        if (args.model_output):
            X_train = train_feat_df.T
            y_train = train_tar_df.T
            clf.fit(X_train, y_train.values.ravel())
            clf.feature_names = row_names

    ###########################################################################

    # Case of test-train prediction (2 datasets OR 1 dataset + model input)
    else:

        # Processing test files into dataframes with parent function "process"
        test_feat_df = process(args.test_feat, test_feat_ext, args.feature, 
                               Marker.FEAT)
        test_tar_df = process(args.test_tar, test_tar_ext, None, Marker.TAR)
        
        # Obtaining feature names and descriptions
        prefilt_row_names = test_feat_df["Name"].tolist()
        prefilt_row_desc = test_feat_df["Description"].tolist()
        row_names = []
        row_desc = []
        # Cleaving them off after
        test_feat_df = test_feat_df.set_index("Name")
        test_feat_df.pop("Description")

        # Instantiating variable to hold value of column names
        #  to use for prediction results
        cols = test_feat_df.columns

        # If no model input is provided
        if (not model_exists):
            # Assigning variables for training feature and target data
            X_train = train_feat_df.T
            y_train = train_tar_df.T

        # Assigning variable for testing feature and target data
        X_test = test_feat_df.T
        y_test = test_tar_df.T

        # If no input model provided
        if (not model_exists):
            # Training the model with training sets of X and y
            # Raveling y_train for data classification format, see last source
            clf.fit(X_train, y_train.values.ravel())
            # No filtering in this case
            row_names, row_desc = prefilt_row_names, prefilt_row_desc
            clf.feature_names = row_names
        else:
            clf = joblib.load(args.model_input)
            # Filtering test data to only include features from fitted model 
            X_test = X_test[clf.feature_names]
            # Doing the same to lists of feature names and descriptions
            for i in range(len(prefilt_row_names)): 
                if prefilt_row_names[i] in clf.feature_names:
                    row_names.append(prefilt_row_names[i])
                    row_desc.append(prefilt_row_desc[i]) 

        if (args.debug and not model_exists):
            print("Training Feature DataFrame: \n", train_feat_df, "\n")
            print("Training Target DataFrame: \n", train_tar_df, "\n\n")
        if (args.debug):
            print("Testing Feature DataFrame: \n", test_feat_df, "\n")
            print("Testing Target DataFrame: \n", test_tar_df, "\n\n")
        
        print(X_test)

        # Predicting using test features
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        # Initializing variable for true target values (test in this case)
        true = test_tar_df.iloc[0].values

        pred_arr = y_pred
        proba_arr = [max(sublist) for sublist in proba]

    ###########################################################################
    
    # If no value was provided for pred_odf filename and feat_odf filename, 
    # use name of feature training file:
    # Not the nicest code, but I think it's much clearer this way (very simple)
    if ((args.pred_odf == None)):
        pred_odf = pred_filename(args.feature)
    else:
        pred_odf = args.pred_odf
        if not pred_odf.endswith(".pred.odf"):
            pred_odf += ".pred.odf"

    if ((args.feat_odf == None)):
        feat_odf = feat_filename(args.feature)
    else:
        feat_odf = args.feat_odf
        if not feat_odf.endswith(".feat.odf"):
            feat_odf += ".feat.odf"
    
    # Removing all /path/before/ (outputs file in curr dir)
    pred_odf = re.sub('.*/', '', pred_odf)
    feat_odf = re.sub('.*/', '', feat_odf)

    # Outputting model file of training data if option is specified
    if (args.model_output):
        if (args.model_output_filename == None):
            joblib.dump(clf, 'model.pkl', compress=('zlib', 3))
        else:
            if not args.model_output_filename.endswith(".pkl"):
                args.model_output_filename += ".pkl"
            joblib.dump(clf, args.model_output_filename, compress=('zlib', 3))

    if (args.debug):
        print("True target values:\n", *true, "\n", sep=" ")
        print("Predicted target values:\n", *pred_arr, "\n", sep=" ")

        # Classifier accuracy check
        accuracy = accuracy_score(true, pred_arr) * 100
        print(f"Accuracy score: " + "{0:.2f}%".format(accuracy), "\n") 

    ###########################################################################

    # Creating pred.odf and feat.odf dataframes
    pred_df = pd.DataFrame(columns=range(true.size))
    feat_df = pd.DataFrame(columns=range(len(proba_arr)))

    # Initializing counter for mismatches
    counter = 0

    if (args.target != None):
        # Initializing array for target name values (e.g, ["all", "aml"])
        tar = tar_array(args.target, train_tar_ext)
    else:
        tar = tar_array(args.test_tar, test_tar_ext)

    if (args.debug):
        print("Target names array:", tar, "\n")

    counter = sum(true != pred_arr)
    value = ["TRUE" if check else "FALSE" for check in true == pred_arr]

    # Assigning true's and preds's values to the respective sample values
    # and evaluating differences. Using tar array to specify target names
    result_list = [[i + 1, list(cols)[i], tar[true[i]], tar[pred_arr[i]],
                    proba_arr[i], value[i]] for i in range(len(cols))]

    # Create DataFrames for pred.odf and feat.odf
    pred_df = pd.DataFrame(result_list, columns=["0", "1", "2", "3", "4", "5"]).T

    # Creating dictionary for pred.odf file header
    pred_header_dict = {
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
        print("Output .pred.odf filename: " + pred_odf)
        if (mode == Marker.TT):
            print("Output .feat.odf filename: " + feat_odf)

    # Passing transposed odf dataframe and header_dict into GP's write_odf()
    write_pred_odf(pred_df.T, pred_odf, pred_header_dict)

    # Only doing feature importance output for test-train mode
    if (mode == Marker.TT):

        feat_df = pd.DataFrame({
            '0': range(1, len(row_names) + 1),
            '1': row_names,
            '2': row_desc,
            '3': clf.feature_importances_
        }).T

        # Creating dictionary for feat.odf file header
        feat_header_dict = {
            "HeaderLines" : "",
            "COLUMN_NAMES" : "Feature Name\t" + "Description\t" + "Count\t",
            "COLUMN_TYPES" : "String \t" + "String\t" + "float\t",
            "Model" : "Prediction Feature Importance",
            "PredictorModel" : "Random Forest Classifier",
            "DataLines" : len(row_names)
        }

        write_feat_odf(feat_df.T, feat_odf, feat_header_dict)

# Otherwise, printing error message to notify user (in dev case CLI-usage)
else:
    print("Please review module instructions.")
