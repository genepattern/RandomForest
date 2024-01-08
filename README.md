# Random Forest
#### Omar Halawa (ohalawa@ucsd.edu) of the GenePattern Team @ Mesirov Lab - UCSD
\
The following repository is a GenePattern module written in Python 3, using the following [Docker image](https://hub.docker.com/layers/genepattern/randomforest/v0.5/images/sha256-5727ecbb059902b1fd6d1e76d2d11b556c4ef8b0c4f193292cd2d965576583d9?context=explore). 
\
It performs either <ins>cross-validation</ins> (takes one dataset as input) or <ins>test-train prediction</ins> (takes two datasets, test and train, with the option of using a pickle file of an already-fitted model instead of the training dataset). Each dataset consists of two file inputs, one for feature data (.gct), and one for target data (.cls). It processes files into DataFrames and performs random forest classification (uses LOOCV (leave-one-out cross validation) in the case of cross-validation) on them using Scikit-learn's RandomForestClassifier, generating an accuracy score, a prediction results file (.pred.odf) that compares the "true" class to the model's prediction, and a feature importance file in the case of test-train prediction. Created for module usage through optional arguments for classifier parameters.

Documentation on usage and implementation is found [here](/docs/tutorial.md).
A detailed step-by-step explanation behind how the Random Forest algorithm works is found [here](/docs/randomforest.md).
All source files, including cross-validation runs for all_aml_train ([.gct](/data/all_aml_train.gct), [.cls](/data/all_aml_train.cls)), BRCA_HUGO ([.gct](/data/all_aml_train.gct), [.cls](/data/all_aml_train.cls)), and iris ([.gct](/data/iris.gct), [.cls](/data/iris.cls)) datasets as well as a test-train run with all_aml_test ([.gct](/data/all_aml_test.gct), [.cls](/data/all_aml_test.cls)) and all_aml_train ([.gct](/data/all_aml_train.gct), [.cls](/data/all_aml_train.cls)) all with [output examples](/data/example_output) ("examples," as the classifier utilizes randomness, so each run varies) are available for better reproducibility and portability. However, to see how randomness can be "reproduced," read [this](/data/example_output/reproduc.md).
