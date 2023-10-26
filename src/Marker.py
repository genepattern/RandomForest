#!/usr/bin/env python3

"""
    Name:          Omar Halawa
    Email:         ohalawa@ucsd.edu
    File name:     Marker.py
    Project:       RandomForest (Non-GPU)
    Description:   Class file that contains the static variables serving as
                   flags for differentiating feature and target data processes
                   as well as what mode the module runs
"""


# Initializing Marker class and its variables for file_valid and mode markers
class Marker:
    FEAT = "FEATURE"
    TAR = "TARGET"
    MODEL = "MODEL"

    TT = "MODULE MODE: TEST-TRAIN PREDICTION"
    LOOCV = "MODULE MODE: LEAVE-ONE-OUT CROSS-VALIDATION"
    FAIL = "ERROR IN FILE INPUTS, MODULE WAS NOT RUN SUCCESSFULLY!"