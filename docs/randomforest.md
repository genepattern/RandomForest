# What is Random Forest?

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Original Paper
Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324

## Introduction

Random Forest is a an [_ensemble_](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/) machine learning algorithm. Specifically, it is an ensemble of **decision trees** (hence the _forest_ part) which basically means that it combines multiple decsision trees in its model. In order to understand Random Forest classification, let us first take one step back and understand what decision trees are.

## Decision Trees

Decision trees are fundamentally quite simple, as they essentially help you answer a classification question (i.e, is what I'm looking at "X or Y" or "Z"). These "X," "Y," and "Z" (or however many you want) labels are called **target** (or **class**) values. In decision trees, all leaf/end nodes hold target values; these are essentially the set of potential results we get upon traversing down the tree. On the other hand, all other nodes are either decision or chance (probability) nodes, both of which split the tree into either more decision/chance nodes or directly toward the final end nodes. However, as the names suggest, chance nodes split the tree into paths that each have an assigned probability (where the sum of all those paths' probabilities sum to 100%) whereas decision nodes split the tree based on a conditional that evaluates a **feature** (or  **data**) value. Feature data is, simply put, a measurment (numerical, like someone's age, or categorical, like their sex) describing an attribute of the objects/samples of interest. See the following decision tree where white nodes are decision nodes and blue & orange nodes are leaf nodes [(image source)](https://www.sciencedirect.com/science/article/pii/S0957417412006707).  

![Decision Tree](https://ars.els-cdn.com/content/image/1-s2.0-S0957417412006707-gr3.jpg)

Given a dataset of feature values and another of target values for the corresponding samples, or objects, we can generate a decision tree. Going back to to the diagram above, as the very simplest case, we can think about this as taking a group of patients of interest, recording their _features_ (age, sex, resting blood pressure, etc..) and seeing what the _target_ value is for each (in this case, do they have diameter narrowing > or < 50%). Then, in order to generate the decision tree using those values, there are a several possible node-splitting criteria we can use which seek to find the best splits using provided feature data and its corresponding class data, the most common of which are the gini criterion and the entropy critertion (see [here](https://quantdare.com/decision-trees-gini-vs-entropy/) for a statistical overview and comparison between them if interested).

Using a decision tree, we are now able to perform **classification**, where when introduced to a new sample that the model (in this case, the decision tree) has never seen before, we can classify that sample using its feature values into one of the target values by simply traversing down the tree.

_Side Note:_ Decision nodes by no means have to be exclusively binary. They could be ternary (a decision node splits into 3 leaves), or they could have as many splits as you want. It all depends on the condition for a specific decision node (is it a simple T/F, or a float range split of zero, positive, or negative?). In fact, any ternary decision tree could be represented as a decision tree, and that applies to any higher-magnitude vs lower-magnitude splitting ([this answer is a nice sanity check for that](https://stats.stackexchange.com/a/12227)). **However**, the reason you may almost always see decision trees as binary is probably due to the combination of T/F binary conditional simplicity of handling feature data and the [technical performance aspect](https://stats.stackexchange.com/questions/12187/are-decision-trees-almost-always-binary-trees) as a split higher than binary can result in an exponentially larger number of nodes, especially if the entire tree is ternary, for example.


## Cons of Decision Trees
With that said, however, decision trees suffer from being highly prone to **overfitting**, behavior characterized by a model accurately predicting its training data but failing to do so with data it has not encountered before (new data, which is the general case). This is a byproduct of many potential issues, but it is primarily due to the primitive nature of individual decision trees in that they will capture the noise of the training data so much so that the tree will end up with extremely specific, unrealistic branches that are almost guaranteed to do more harm than good in general classification cases. This high senitivity to the training data can lead to variance. This can be somewhat prevented with setting a maximum tree-depth. However, doing so dilutes the predictive power of a decision tree and introduces another issue, error due to bias, as the tree may not be going as deep as it should and would therefore be making more biased decisions (see [here](https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991)).


## Random Forest
As such, having to balance the risk of overfitting and not wanting to compensate with a much weaker, less accurate predictive model, individual decision trees are not ideal for the purposes of classification. However, through a large ensemble of slightly different decision trees , we can minimize risk of overfitting greatly while maintaining a strong predictive classification model. This is the core concept of Random Forest - an ensemble of decision trees that utilizes **bagging** (or **bootstrapping and aggregation**) as well as **random feature selection**.

* Bagging: An ensemble technqiue that utilizes bootstrapping and aggregation (done in that order).
    * Bootstrapping: This is when each decision tree is trained on a random subset of the samples while still maintaining the same number of samples as the original (due to sample selection being done one at a time, this allows repeats which is perfectly fine)
    * Aggregation: This is when, during classification, we carry out the prediction for each decision tree and perform majority voting on the output target value, yielding the final classification

* Random Feature Selection: For each decision tree, with regards to feature data, two things must occur: 
    1. A random subset of features is selected for each decision tree
    2. The number of features for each decision tree is the same, and for optimal splitting, is typically sqrt(number of features) or log(number of features) (see [here](https://link.springer.com/article/10.1007/s10994-006-6226-1) for why)

_Note:_ These techniques that make Random Forest what it is can of course be altered by the user, but that would probably defeat the purpose of the Random Forest classifier.

See the following image which represents the entire process of Random Forest Classification [(image source)](https://catalyst.earth/catalyst-system-files/help/concepts/focus_c/oa_classif_intro_rt.html).

![Random Forest](https://catalyst.earth/catalyst-system-files/help/COMMON/references/images/RT_schematic.png)


## Leave-One-Out Cross Validation
For the module's purposes of performing classification on an input feature dataset, a test/train split would not work as that would bring about **data leakage** through testing with the same data that the model has been trained on. Therefore, in order to achieve random forest classification on a feature dataset as the only provided input, we can perform Leave-One-Out Cross Validation (LOOCV). Essentially, for each sample of the data, we perform Random Forest classification using every other sample. This allows us to obtain a predicted target value for every sample, giving us the desired Random Forest classification. See the following [(image source)](https://dataaspirant.com/7-loocv-leave-one-out-cross-validation/):

![LOOCV](https://dataaspirant.com/wp-content/uploads/2020/12/7-LOOCV-Leave-One-Out-Cross-Validation.png)

## Motivation
Random Forest classification is a very powerful and diverse tool for not only machine learning technologies but also medical and bioinformatics applications as well. It reaps the benefits of decision trees while minimizing their flaws, and its only real shortcoming is in being very computationally-demanding. However, with future [SDSC](https://www.sdsc.edu/) implementation, a GPU Random Forest classifier will not only serve as a powerful tool for all kinds of biologists and researchers, but it will also be a great entry point into further GPU-centric GenePattern modules as well as a more solid standardization of GPU-centric module development.

