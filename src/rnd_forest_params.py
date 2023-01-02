#!/usr/bin/env python3

# Importing modules
import argparse as ap
import math
import json

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     rnd_forest_params.py
    Project:       RandomForest (Non-GPU)
    Description:   Non-GPU RandomForest python script that handles argument
                   parsing from command line. Contains two required arguments
                   for file input (classifier features and classifier target).
                   Also contains optional arguments of random forest classifier
                   and debugging. Has Range class to restrict argument input as
                   well as helper methods for specific argument input types.
    References:    tiny.cc/rnd_forest_dict
                   tiny.cc/scikit_rnd_forest
                   tiny.cc/arg_none_or_str
                   tiny.cc/arg_range
"""

# NOTE: As this program is entirely intended for module development 
#       on GenePattern Dev which allows for range-limiting on argument inputs,
#       certain value range errors are not handled as elegantly as could be
#       (seeing as they are impossible to get when using the module's features)
#
#       For non-module implementation, handling such out-of-range errors could
#       be best approached through overriding the error method in Argparse 


# Classs to limit possible argument values of type int & float (precise range)
# (see last source)
class Range(object):
    # Initializer with self-reference, start value, and end value
    def __init__(self, start, end):
        self.start = start
        self.end = end

    # Class method to overload == operator to compare class objects
    def __eq__(self, other):
        # Accounting for cases where arguments can be None or an int/float 
        if (other == None):
            return None
        # Else, carrying on comparison
        else:
            return self.start <= other <= self.end

# Bound-excluding version, see the above Range class for details  
# Note: only used for the test_size argument, could not find better solution
class Exc_Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        if (other == None):
            return None
        else:
            return self.start < other < self.end

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
# Only affects non-module usage
def cli_bool(value):
    # Checking for string input ("True" or "False")
    if (type(value) == str):
        # Returning boolean counterpart
        return eval(value)
    else:
        return value


# Adding arguments to script for classifier feature file and target file input,
# scikit RandomForest classifier parameters, and debugging
parser = ap.ArgumentParser(description='Scikit Random Forest Classifier')


# Adding file input arguments (required)
# Feature file input (.gct):
parser.add_argument("-f", "--feature", help="classifier feature data filename"
                    + " Valid file format(s): .gct", required=True)
# Target file input (.cls):
parser.add_argument("-t", "--target", help="classifier target data filename"
                    + " Valid file format(s): .cls", required=True)


# Program debug argument, either True or False, False by default
parser.add_argument("-d", "--debug", help="output program debug messages",
                    nargs="?", const=1, default=False, type=cli_bool)

# Test/Training set split argument, 30% for test is default (70% for training):
parser.add_argument("--test_size", 
                    help="ratio for test data split, rest is training data",
                    nargs="?", const=1, default=0.3, type=float,
                    choices=[Exc_Range(0.0, 1.0)])


# Random Forest Classifier arguments (optional) as default values exist for all
parser.add_argument("--bootstrap", help="boolean for bootstrapping",
                    nargs="?", const=1, default=True, type=cli_bool,
                    choices=[True, False])

parser.add_argument("--ccp_alpha", 
                    help="complexity parameter of min cost-complexity pruning",
                    nargs="?", const=1, default=0.0, type=float,
                    choices=[Range(0.0, math.inf)])

# TODO, is dict/lis of, but generally,is "balanced" or "balanced_subsample"
parser.add_argument("--class_weight", help="class weight specification",
                    nargs="?", const=1, default=None, type=none_or_str)

parser.add_argument("--criterion", help="criterion of node splitting",
                    nargs="?", const=1, default="gini", type=str)

parser.add_argument("--max_depth", help="maximum tree depth",
                    nargs="?", const=1, default=None, type=none_or_int,
                    choices=[Range(1, math.inf), None])

# TODO, can be "sqrt", "log2"("auto" removed in 1.3), a float, or an int
parser.add_argument("--max_features", 
                    help="number (ratio in cuML) of features per split",
                    nargs="?", const=1, default="sqrt", type=str)

parser.add_argument("--max_leaf_nodes", help="maximum leaf nodes per tree",
                    nargs="?", const=1, default=None, type=none_or_int,
                    choices=[Range(2, math.inf), None])

parser.add_argument("--max_samples", 
                    help="number (ratio for cuML) of datasets to use per tree",
                    nargs="?", const=1, default=None, type=none_or_float,
                    choices=[Range(0.0, 1.0), None])

parser.add_argument("--min_impurity_decrease", 
                    help="minimum impurity decrease needed per node split",
                    nargs="?", const=1, default=0.0, type=float,
                    choices=[Range(0.0, math.inf)])

# Using integer implementation [1, inf) and NOT float implementation (0.0, 1.0)
parser.add_argument("--min_samples_leaf", 
                    help="minimum number of samples required at leaf node",
                    nargs="?", const=1, default=1, type=int,
                    choices=[Range(1, math.inf)])

# Using integer implementation [2, inf) and NOT float implementation (0.0, 1.0)
parser.add_argument("--min_samples_split", 
                    help="minimum sample number to split node",
                    nargs="?", const=1, default=2, type=int,
                    choices=[Range(2, math.inf)])

parser.add_argument("--min_weight_fraction_leaf", 
                help="min weighted fraction of weight sum total to be leaf",
                nargs="?", const=1, default=0.0, type=float,
                choices=[Range(0.0, 0.5)])

parser.add_argument("--n_estimators", 
                    help="number of trees in forest",
                    nargs="?", const=1, default=100, type=int,
                    choices=[Range(1, math.inf)])

# Note, n_jobs=0 has no meaning in Random Forest Classifier, -1 for all CPUs
parser.add_argument("--n_jobs", 
                    help="number of parallel streams for building the forest",
                    nargs="?", const=1, default=None, type=none_or_int,
                    choices=[Range(-1*math.inf, -1), Range(1, math.inf), None])

parser.add_argument("--oob_score", 
                    help="if out-of-bag samples used for generalization score",
                    nargs="?", const=1, default=False, type=cli_bool,
                    choices=[True, False])

# Random Forest Classifier has seed value limit at 2^32 - 1, this is 4294967295
max_seed = (2**32) - 1
parser.add_argument("--random_state", 
                    help="seed for random number generator",
                    nargs="?", const=1, default=None, type=none_or_int,
                    choices=[Range(0, max_seed), None])

# 0 for no verbosity, 1 for basic verbosity, values greater for more verbosity
parser.add_argument("--verbose", help="verbosity flag",
                    nargs="?", const=1, default=0, type=int, 
                    choices=[Range(0, math.inf)])

parser.add_argument("--warm_start", 
                    help="whether to start new forest or add to past solution",
                    nargs="?", const=1, default=False, type=cli_bool,
                    choices=[True, False])

# Parsing arguments for future calls within script to utilize
args = parser.parse_args()