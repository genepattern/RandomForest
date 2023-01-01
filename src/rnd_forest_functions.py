#!/usr/bin/env python3

# Importing from current directory files
from rnd_forest_params import *
from Marker import *
from Extension import *

# Importing modules
import pandas as pd
from os.path import exists

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     rnd_forest_functions.py
    Project:       RandomForest (Non-GPU)
    Description:   Non-GPU RandomForest python script that contains all file
                   handling, Random Forest Classifier logic, and argument value
                   getter functions which are used in the main script and the
                   Classifier class file.
"""


def file_valid(name, marker):
    """ Function that checks for filename validity through existence & format.
        Has general format to allow for future file type implementation.       

    Arguments:
        name:   filename (with extension) to check existence of in current dir
        marker: String value indicating whether file is of features or targets
                (check Marker class)
    Returns:
        ext:  a String of the file extension if it passes name and ext check
        None: if the file name or extension are not valid
    """

    # Intializing placeholder for proper file extension list
    curr_list = None
    # Initalizing placeholder for file extension
    curr_ext = None

    # Validating existent file name
    valid_name = exists(name)

    # Intializing extension check as False by default
    valid_ext = False

    # Validating file extension through checking marker value (by Marker class)
    # Then assigning curr_list's value to one of the two Extension class lists
    # Only two cases possible, calls are within program
    
    # Feature data file case
    if (marker == Marker.FEAT):
        curr_list = Extension.FEAT_EXT
    # Target data file case
    elif (marker == Marker.TAR):
        curr_list = Extension.TAR_EXT

    # Carrying out the extension check logic using the now-updated curr_list
    for ext in curr_list:
        if name.endswith(ext):
            # If match is found, updating valid_ext to True
            valid_ext = True
            # Also assigning curr_ext's value
            curr_ext = ext
            # Breaking once match is found
            break

    # Accounting for both checks 
    if (valid_name and valid_ext):
        # if checks are valid, returning file extension
        return curr_ext
    else:
        # Invalid file name message
        if (not valid_name):
            print("File name '" + name + 
                "' is invalid (not found in current directory).")
        # Invalid file extension message
        if (not valid_ext):
            # Note: Future file type implementation must list all valid
            # extensions for classifier input or target, respectively 
            print("File extension of '" + name + "' is invalid. " + 
                "Expected extension(s): ", end="")
            print(curr_list)
        
        print()
        # Returning None in the case of invalid file name
        return None


def get_feature():
    """ Function that returns name of the file input for classifier feature(s)
    Returns:
        args.feature: a String of the classifier feature data file's name
    """

    # Returning the classifier feature data file name
    return args.feature
        

def get_target():
    """ Function that returns name of the file input for classifier target(s)
    Returns:
        args.target: a String of the classifier target data file's name
    """

    # Returning the classifier target data file name
    return args.target


def process(name, ext, marker):
    """ Function that processes valid file given its extension as an argument

    Arguments:
        name:   name of file to process
        ext:    extension of file to process (obtained from file_valid call)
        marker: String value indicating whether file is of features or targets
                (only in cases of a file type that can be both target & marker)
    Returns:    returns a call to the appropriate helper function that contains
                actual logic for processing
    """

    if (ext == Extension.GCT_EXT):
        return gct_process(name)
    elif (ext == Extension.CLS_EXT):
        return cls_process(name)
    # elif (ext == Extension.TXT_EXT):
    #     return txt_process(name,marker)


def gct_process(name):
    """ Function that processes .gct file of feature data to a pandas DataFrame

    Argument:
        name:       name of .gct file to process into df
    Returns:
        gct_file:   processed pandas df of the .gct feature data file
    """

    # Creating df of features from classifier feature data file (.gct process)
    gct_file = pd.read_csv(name, skiprows = 2, sep = '\t')

    # Removing non-biological data columns
    gct_file.pop("Name")
    gct_file.pop("Description")

    # Returning processed df
    return gct_file


def cls_process(name):
    """ Function that processes .cls file of target data to a pandas DataFrame

    Argument:
        name:       name of .cls file to process into df
    Returns:
        cls_file:   processed pandas df of the .cls target data file
    """

    # Creating df of target(s) from classifier target data file (.cls process)
    cls_file = pd.read_csv(name, skiprows = 2, sep = '\s+', 
                            header=None)
    # Returning processed df
    return cls_file


# def txt_process(name, marker)
#     return None


def get_debug():
    """ Function that returns truth value of debug argument
    Returns:
        args.debug: a Boolean value of whether debugging is on
    """

    # Returning the Boolean value
    return args.debug


# Scikit Random Forest Classifier Getter Methods:
def get_bootstrap():
    """ Function that returns truth value of boostrap argument
    Returns:
        args.bootstrap: a Boolean value of whether boostrapping is on
    """

    # Returning the Boolean value
    return args.bootstrap


def get_ccp():
    """ Function that returns the float value of ccp_alpha argument
    Returns:
        args.ccp_alpha: float value of ccp_alpha
    """

    # Returning the float value
    return args.ccp_alpha


def get_class_weight():
    """ Function returning the dict (or list of) value of class_weight argument
    Returns:
        args.class_weight: json.loads dict (or list of) value of class_weight
    """

    # Returning the json.loads dict (or list of) value
    return args.class_weight


def get_criterion():
    """ Function returning the String value of the criterion argument
    Returns:
        args.criterion: String value of the criterion argument
    """

    # Returning the string value
    return args.criterion


def get_max_depth():
    """ Function returning the integer value of the max_depth argument
    Returns:
        args.max_depth: Integer value of the max_depth argument
    """

    # Returning the integer value
    return args.max_depth


def get_max_features():
    """ Function returning the String/float value of the max_features argument
    Returns:
        args.max_features: String/float value of the max_features argument
    """

    # Returning the String/float value
    return args.max_features


def get_max_nodes():
    """ Function returning the integer value of the max_leaf_nodes argument
    Returns:
        args.max_leaf_nodes: Integer value of the max_leaf_nodes argument
    """

    # Returning the integer value
    return args.max_leaf_nodes


def get_max_samples():
    """ Function returning the float value of the max_samples argument
    Returns:
        args.max_samples: Float value of the max_samples argument
    """

    # Returning the float value
    return args.max_samples


def get_impurity():
    """ Function returning float value of the min_impurity_decrease argument
    Returns:
        args.min_impurity_decrease: Float value of min_impurity_decrease
    """

    # Returning the float value
    return args.min_impurity_decrease


def get_min_samples_leaf():
    """ Function returning the integer value of the min_samples_leaf argument
    Returns:
        args.min_samples_leaf: Integer value of the min_samples_leaf argument
    """

    # Returning the integer value
    return args.min_samples_leaf


def get_split():
    """ Function returning the integer value of the min_samples_split argument
    Returns:
        args.min_samples_split: Integer value of the min_samples_split argument
    """

    # Returning the integer value
    return args.min_samples_split


def get_min_weight_fraction():
    """ Function returning float value of the min_weight_fraction_leaf argument
    Returns:
        args.min_weight_fraction_leaf: Float value of min_weight_fraction_leaf
    """

    # Returning the float value
    return args.min_weight_fraction_leaf


def get_estimators():
    """ Function returning the integer value of the n_estimators argument
    Returns:
        args.n_estimators: Integer value of the n_estimators argument
    """

    # Returning the integer value
    return args.n_estimators


def get_jobs():
    """ Function returning the integer value of the n_jobs argument
    Returns:
        args.n_jobs: Integer value of the n_jobs argument
    """

    # Returning the integer value
    return args.n_jobs


def get_oob():
    """ Function returning the boolean value of the oob_score argument
    Returns:
        args.oob_score: boolean value of the oob_score argument
    """

    # Returning the boolean value
    return args.oob_score


def get_random():
    """ Function returning the integer value of the random_state argument
    Returns:
        args.random_state: Integer value of the random_state argument
    """

    # Returning the integer value
    return args.random_state


def get_verbose():
    """ Function returning the integer value of the verbose argument
    Returns:
        args.verbose: Integer value of the verbose argument
    """

    # Returning the integer value
    return args.verbose


def get_warm():
    """ Function returning the boolean value of the warm_start argument
    Returns:
        args.warm_start: boolean value of the warm_start argument
    """

    # Returning the boolean value
    return args.warm_start