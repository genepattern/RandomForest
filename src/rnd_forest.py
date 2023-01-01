#!/usr/bin/env python3

# Importing getter and processing functions
from rnd_forest_functions import *
# Importing Classifier class to instantiate a Classifier object
from Classifier import *
# Importing Marker class to differentiate between feature & target files/logic
from Marker import *

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     rnd_forest.py
    Project:       RandomForest (Non-GPU)
    Description:   Non-GPU RandomForest main python script that contains the
                   parent calls to: take in and validate files for classifier 
                   feature (.gct) and target (.cls) data inputs, instantiate a
                   Classifier object to process the files as dataframes, and
                   perform the Random Forest classification as well as predict
                   accuracy. All logic is found in the other files.
                   Created for module integration on GenePattern Dev.
                   Designed to allow for further file type implementation.
"""

# Verifying debug status
if (get_debug()):
    print("Debugging on.")
    print()

# Checking for feature and target data file validity by calling file_valid
# Obtained output corresponds to file extension if valid, None if invalid
feature_ext = file_valid(get_feature(), Marker.FEAT)
target_ext = file_valid(get_target(), Marker.TAR)

# Only carrying out Random Forest Classification if both files are valid
if ((feature_ext != None) and (target_ext != None)):

    # Processing the valid files into dataframes with parent function "process"
    feature_df = process(get_feature(), feature_ext, Marker.FEAT)
    target_df = process(get_target(), target_ext, Marker.TAR)

    # Creating an instance of the Classifier object as rnd_forest
    rnd_forest = Classifier(feature_df, target_df)

# Otherwise, printing error message to notify user
else:
    print("Error in file input, please check above for details.")


