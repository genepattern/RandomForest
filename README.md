# Random Forest Classifier (Non-GPU)
#### Omar Halawa (ohalawa@ucsd.edu) of the GenePattern Team @ Mesirov Lab - UCSD
\
The following repository is a GenePattern module written in Python 3, using the following [Docker image](https://hub.docker.com/layers/omarhalawa/randomforest/1.0/images/sha256-995d424aa0fa77f608aaa5575faafad6cea966a377fdb8dd51e9144e74f7ff21?context=repo). 
\
It serves as a foundation for implementing the cuML-based GPU Random Forest Classifier.

It takes in two files, one for feature data (.gct), and one for target data (.cls). Then, it processes them into DataFrames and performs random forest classification via LOOCV ([leave-one-out cross validation](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85)) on them using Scikit-learn's RandomForestClassifier, generating an accuracy score and a target prediction done on the entire feature dataset and outputting a prediction results (.pred.odf) file. Created for module usage through optional arguments for classifier parameters. Designed for smooth implementation of other file type inputs and future features.

Documentation on usage and implementation is found [here](https://github.com/omarhalawa3301/randomforest/blob/main/docs/tutorial.md).
All source files, including all_aml ([.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.gct), [.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.cls)), BRCA_HUGO ([.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.gct), [.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/all_aml_train.cls)), and iris ([.gct](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.gct), [.cls](https://github.com/omarhalawa3301/randomforest/blob/main/data/iris.cls)) test datasets with [output examples](https://github.com/omarhalawa3301/randomforest/blob/main/data/example_output) ("examples," as the classifier utilizes randomness, so each run varies) are available for better reproducibility and portability. 
