#!/usr/bin/env python3

# Importing classes to differentiate feature & target logic and extensions
from Marker import *
from Extension import *

# Importing modules
import pandas as pd
from os.path import exists

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     rnd_forest_functions.py
    Project:       RandomForest (GPU)
    Description:   GPU RandomForest python script that contains file
                   handling functions which are used in the main script.
    References:    github.com/genepattern/genepattern-python/blob/master/gp/
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

    # Checking for case of no provided input (test dataset input)
    if (name == None):
        return None

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


def process(name, ext, train_df, marker):
    """ Function that processes valid file given its extension as an argument

    Arguments:
        name:   name of file to process
        ext:    extension of file to process (obtained from file_valid call)
        train_df:   Training feature filename, provided only if processing Test 
                    feature data, else is None (needed for gct_process)
        marker: String value indicating whether file is of features or targets
                (only in cases of a file type that can be both target & marker)
    Returns:    returns a call to the appropriate helper function that contains
                actual logic for processing
    """

    if (ext == Extension.GCT_EXT):
        return gct_process(name, train_df)
    elif (ext == Extension.CLS_EXT):
        return cls_process(name)
    # elif (ext == Extension.TXT_EXT):
    #     return txt_process(name,marker)


def gct_process(name, train_df):
    """ Function that processes .gct file of feature data to a pandas DataFrame

    Argument:
        name:       name of .gct file to process into df
        train_df:   Training feature filename, provided only if processing Test 
                    feature data, else is None
    Returns:
        gct_file:   processed pandas df of the .gct feature data file
    """
    # Creating df of features from classifier feature file (.gct process)
    gct_file = pd.read_csv(name, skiprows = 2, sep = '\t')

    # Making sure to sort data by the required "Names" column, alphabetical 
    gct_file.sort_values("Name")
    
    # Processing if working with test and train datasets (not LOOCV)
    if (train_df != None):

        # Creating df of the training data
        train = pd.read_csv(train_df, skiprows = 2, sep = '\t')

        # Getting intersection the two dataframes via their common row names
        gct_file = gct_file[gct_file.Name.isin(train.Name)]

        # Resetting index after the intersection
        gct_file = gct_file.reset_index(drop=True)

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


def write_odf(df, path, headers):
    """ Function to write out result .pred.odf file and creates its header str

    Argument:
        df:         DataFrame for file to contain
        path        path to output file to (and name)
    Returns:
        Nothing, just outputs file
    """

    # Add the initial ODF version line
    head = 'ODF 1.0\n'

    # Add HeaderLines
    head += 'HeaderLines=' + str(len(headers)-1) + '\n'
    head += 'COLUMN_NAMES:\t' + str(headers['COLUMN_NAMES']) + '\n'
    head += 'COLUMN_TYPES:\t' + str(headers['COLUMN_TYPES']) + '\n'
    head += 'Model=' + str(headers['Model']) + '\n'
    head += 'PredictorModel=' + str(headers['PredictorModel']) + '\n'
    head += 'NumFeatures=' + str(headers['NumFeatures']) + '\n'
    head += 'NumCorrect=' + str(headers['NumCorrect']) + '\n'
    head += 'NumErrors=' + str(headers['NumErrors']) + '\n'
    head += 'DataLines=' + str(headers['DataLines']) + '\n'

    # Processes headers using gp method and and writes out file, see reference 
    with open(path, 'w') as file:
        file.write(head)
        df.to_csv(file, sep='\t', header=False, index=False, mode='w+')


def pred_filename(name):
    """ Function that takes in a full filename and returns it as a string
        without its extension. Uses feature file by default. 

    Argument:
        name:       name of target file to cut extension off of
    Returns:
        filename:   processed pandas df of the .cls target data file
        None:       if no match
    """
    
    # Checking if the extension is one that exists in the array of feature exts
    # Processing end of feature file
    # Splitting at final occurence of a dot for both cases
    feature_end = "." + name.rsplit(".", 1)[1]
    feature_name = name.rsplit(".", 1)[0]

    # Iterating through all feature file extensions
    for ext in Extension.FEAT_EXT:

        # Checking for match
        if (feature_end == ext):
            # Returning pre-extension str if so
            return (feature_name + Extension.ODF_EXT)

        else:
            return None

    
def tar_array(name, ext):
    """ Function that takes in a a target data filename and its extension
        and returns an array of all possible target values in the same order. 

    Argument:
        name:        name of target file
         ext:        ext of target file
    Returns:
         tar:        ordered array of all possible target
    """
    
    # If statement for future file format implementation
    if (ext == Extension.CLS_EXT):

        # Processing file's 2nd line by reading it in, stripping newline char,
        # and removing first character ("#")
        tar_file = open(name, "r")
        tar_file.readline()
        tar = tar_file.readline().strip('\n')
        
        # Removing any string of spaces (tab or whitespace) as .cls head delim
        tar = tar.split(sep=None)
        tar.__delitem__(0)

        # Returning the array of target values
        return tar
