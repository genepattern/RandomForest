The following are output examples of the test datasets in the upper directory ("examples," as the classifier utilizes randomness, so each run varies) are available for better reproducibility and portability.

# Note: (output examples are of random_state=64)
**In the Scikit, Non-GPU version of the module**, setting the random_state arugment to a specific integer, like 64, will always yield the same prediction results and output file because this is the "seed" that dictates how, for a specific dataset input (data file and class file), bagging and random feature selection (the random processes) are carried out.

# However, RAPIDS.ai does not currently fully guarantee the exact same results using the same seed!
