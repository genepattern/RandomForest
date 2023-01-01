# Random Forest Classifier (Non-GPU)
#### Omar Halawa (ohalawa@ucsd.edu) of GenePattern Team @ Mesirov Lab - UCSD
\
The following repository is a GenePattern module written in Python 3, using the following [Docker image](https://hub.docker.com/layers/genepattern/notebook-python39/22.04/images/sha256-1182e33d0a4d944e676003b2d4a410ec3a197db13847292cedca441a0541513d?context=explore). 

It takes in two , validates and then processes them into DataFrames (turning all non-positive values into 0), and outputs a new GCT file of the processed data. Documentation on usage and implementation is found [here](https://github.com/omarhalawa3301/randomforest/blob/main/docs/tutorial.md).

All source files, including two s [sample test file](https://github.com/omarhalawa3301/log_normalize/blob/main/data/all_aml_train.gct) and [expected output file](https://github.com/omarhalawa3301/log_normalize/blob/main/data/result.gct), are available for better reproducibility and portability. 
