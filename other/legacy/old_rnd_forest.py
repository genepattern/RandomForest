#!/usr/bin/env python3

# Importing necessary functions and the Extension class
from rnd_forest_functions import *

# Importing modules
import pandas as pd
import argparse as ap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     old_rnd_forest.py
    Project:       RandomForest (Non-GPU)
    Description:   Non-GPU RandomForest python script that contains the logic
                   for taking in feature (.gct) and target (.cls) classifier 
                   data files, processing them through pandas DataFrames, 
                   and applying the random forest classification. 
                   Created for module integration on GenePattern Dev.
                   Designed to allow for more file type implementation.
    References:    tiny.cc/scikit_rnd_forest
                   datacamp.com/tutorial/random-forests-classifier-python
                   scholarworks.utep.edu/cs_techrep/1209/
                   tiny.cc/7cl2vz
"""

# Verifying verbosity status
if (get_verbose()):
    print("Verbosity on.")
    print()

# Checking for input files' validity (.gct and .cls)
gct_valid = file_valid(get_gct(), Extension.GCT_EXT)
cls_valid = file_valid(get_cls(), Extension.CLS_EXT)

if (gct_valid and cls_valid):

    # Creating df of features from classifier feature data file (.gct process) 
    gct_file = pd.read_csv(get_gct(), skiprows = 2, sep = '\t')
    # Removing non-biological data columns
    gct_file.pop("Name")
    gct_file.pop("Description")

    # Creating df of target(s) from classifier target data file (.cls process)
    cls_file = pd.read_csv(get_cls(), skiprows = 2, sep = '\s+', header=None)

    # Split dataset into training set and test set
    # 70% training and 30% test (to know why check ScholarWorks references)
    X_train, X_test, y_train, y_test = train_test_split(gct_file.T,
                                                    cls_file.T, test_size=0.3)

    # Creating instance of RandomForestClassifier
    clf = RandomForestClassifier(bootstrap=True, class_weight=None, 
                criterion='gini', max_depth=None, max_features='sqrt', 
                max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=100, n_jobs=None, oob_score=False, random_state=None,
                verbose=0, warm_start=False)

    # Training the model with training sets of X and y
    # Raveling y_train's values for data classification format, see last reference
    clf.fit(X_train, y_train.values.ravel())

    # Predicting 
    y_pred=clf.predict(X_test)

    # Classifier accuracy check
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Example of predict, check result
    print(clf.predict(X_train))
