# Random Forest Classifier (Non-GPU)
#### Omar Halawa (ohalawa@ucsd.edu) of GenePattern Team @ Mesirov Lab - UCSD
\
The following repository is a GenePattern module written in Python 3, using the following [Docker image](https://hub.docker.com/layers/genepattern/notebook-python39/22.04/images/sha256-1182e33d0a4d944e676003b2d4a410ec3a197db13847292cedca441a0541513d?context=explore). 
\
Serves as foundation for cuML-based GPU Random Forest Classifier.

It takes in two files, one for classifier feature data (.gct), and one for classifier target data (.cls). Then, it processes them into DataFrames and performs classification on them using Scikit-learn's RandomForestClassifier, generating an accuracy score. Created for both command-line and module usage through optional arguments for classifier parameters. Designed for smooth implementation of other file type inputs and future features. 

Documentation on usage and implementation is found [here](https://github.com/omarhalawa3301/randomforest/blob/main/docs/tutorial.md).
All source files, including iris ([.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.gct), [.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.cls)) and all_aml ([.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.gct), [.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.cls)) test datasets with output examples ("examples," as the classifier utilizes randomness, so each run varies) are available for better reproducibility and portability. 
