# What is Random Forest?

**Author**: Omar Halawa, GenePattern Team @ Mesirov Lab - UCSD

**Contact**: [Email](mailto:ohalawa@ucsd.edu)

## Introduction

Random Forest is a an [_ensemble_](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/) machine learning algorithm. Specifically, it is an ensemble of **decision trees** (hence the _forest_ part) which basically means that it combines multiple decsision trees in its model. In order to understand Random Forest classification, let us first take one step back and understand what decision trees are.

## Decision Trees

Decision trees are fundamentally quite simple, as they essentially help you answer a classification question (i.e, "Is it X or Y" or "Z"). These "X," "Y," and "Z" (or however many you want) labels are called **target** (or **class**) values. In decision trees, all leaf/end nodes hold target values; these are essentially the set of potential results we get upon traversing down the tree. On the other hand, all other nodes are either decision or chance (probability) nodes, both of which split the tree into either more decision/chance nodes or directly toward the final end nodes. However, as the names suggest, chance nodes split the tree into paths that each have an assigned probability (where the sum of all the paths' probabilities sum to 100%) whereas decision nodes split the tree based on a conditional that evaluates a **feature** (or  **data**) value. See the following decision tree as a sanity check (it contains no chance nodes).  

![Decision Tree](https://miro.medium.com/max/1400/1*jojTznh4HOX_8cGw_04ODA.png)
[Source](https://heartbeat.comet.ml/understanding-the-mathematics-behind-decision-trees-22d86d55906)

Given a dataset of feature values and another of target values for the corresponding samples, or objects, we can generate a decision tree. To do so, there are a variety of approaches. 

_Note:_ Decision nodes by no means have to be exclusively binary. They could be ternary (a decision node splits into 3 leaves), or they could have as many splits as you want. It all depends on the condition for a specific decision node (is it a simple T/F, or a float range split of zero, positive, or negative?). In fact, any ternary decision tree could be represented as a decision tree, and that applies to any higher-magnitude vs lower-magnitude splitting ([this answer is a nice sanity check for that](https://stats.stackexchange.com/a/12227)). **HOWEVER**, the reason you may almost always see decision trees as binary is probably due to the [technical performance asepct](https://stats.stackexchange.com/questions/12187/are-decision-trees-almost-always-binary-trees) as a split higher than binary results in an exponentially larger number of nodes (depending on where the ternary split occurs), especially if the entire tree is ternary, for example.


## Decision Tree Example
* [The GenePattern RandomForest source repository](https://github.com/omarhalawa3301/randomforest)
* RandomForest uses the [omarhalawa/randomforest:1.0](https://hub.docker.com/layers/omarhalawa/randomforest/1.0/images/sha256-995d424aa0fa77f608aaa5575faafad6cea966a377fdb8dd51e9144e74f7ff21?context=repo) docker image

## Motivation
This module only requires feature (.gct) and target (.cls) classifier data files as well as an output filename as user-input. Other parameters are optional, maintaining default values if left unchanged (see below).

## Required Inputs

1. data file  
    This is the input file of classifier feature data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a GCT file (.gct), but future support for other feature data formats will be implemented.  
      
2. cls file  
    This is the input file of classifier target data which will be read in by the python script and ultimately will be processed through random forest classification. The parameter expects a CLS file (.cls), but future support for other feature data formats will be implemented.  

## Miscellaneous
