# Optimizing an ML Pipeline in Azure

## Overview
Problem Statement: To analyze the provided bankmarketing dataset and predict whether the client will subscribe for a bank term deposit based on the best model predictions.
In this project, Aure ML pipeline is built using the provided bankmarketing dataset through Auto-ML to compare and optimize the results of the best model through Hyperdrive against the custom-coded model—a standard Scikit-learn Logistic Regression.

Pipeline Architecture

![image](https://user-images.githubusercontent.com/46094082/211244605-d1a15f6a-11c4-4b50-86d2-2b18822b6f83.png)


https://learn.udacity.com/nanodegrees/nd00333/parts/cd0600/lessons/fe72a17d-091f-4c9c-b341-d2fea440a791/concepts/5632c8e6-e3ac-4873-a9b5-b1164387e6fb

Key Steps of the pipeline:

- Create tabular data set.
- Split dataset into train and test sets (Split Ratio 80:20).
- Specify Parameter Sampler.
- Specify Policy.
- Using HyperDrive config.
- Using AutoML configuration set AutomML config.
- Submit experiment to find the best fit model.



## Summary
Dataset: The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

For this classification task of predicting, ‘yes’ or ‘no’, the Voting Ensemble model from the AutoML pipeline emerged as the best forming model with an accuracy of 91.5%.

<img width="1031" alt="Auto-ML Best Model" src="https://user-images.githubusercontent.com/46094082/211244669-e532339c-7be8-4b3a-ad41-273e20db3eb0.png">



## Scikit-learn Pipeline
In this project, Azure ML SDK services is used to train a standard Scikit-learn Logistic Regression model on the Bank Marketing ‘tabular’ dataset on a single-node CPU leveraging the capabilities of the Azure ML HyperDrive for optimizing the hyperparameters.
Scikit-learn pipeline gave an accuracy of **0.91183 HyperDrive Model

<img width="1029" alt="Scikit-learn Pipeline" src="https://user-images.githubusercontent.com/46094082/211244695-4ffebfa9-c161-498a-bb96-0ae29d98715b.png">


The key steps for HyperDrive Tuning involves search space,Sampling Method,Primary metric and early termination policy. In this project Random Sampling search space adopted with the intention to randomly sample hyperparameter values from a defined search space without incurring high computational cost.Even though search space supports both discrete and continuous values as hyperparameter values but the search space is set to discrete for Regularization parameter, C, and Max-iter because it achieved the best accuracies compared to the accuracies obtained from the model when the continuous search space was used.

## Parameter sampler

Selection for Random Sampling expressed as:

Parameter sampling
Sampling policy name

RANDOM
Parameter space
{"--C":["choice",[[0.001,0.01,0.1,1,10]]],"--max_iter":["choice",[[100,200,300]]]}


Random parameter sampling is quicker and gives predictions close to accurate but if compute cost is not a concern, GridParameter Sampling is most accurate but very exhaustive.
 

## Stopping Policy
Random search space supports early termination of low-performing models. To apply the early stopping policy, this project adopted the “Bandit Termination Policy” to ensure that the Azure ML pipeline does not waste time exploring runs with hyperparameters that are not promising.

Terminating poorly performing model is extremly important when considering time and compute aspects. If the error rate is not with in normal range then it should be terminated. The policy is expressed as:


policy = BanditPolicy(evaluation_interval=2, slack_factor=0.12)

*slack facor specifies ratio to which model should allow before terminating.* 

T= Metric/((1+S) ) where T is the termination threshold,S,the slack-factor.

A run terminates when metric < T.
 
 
## AutoML
An AutoML is built on the Bank Marketing dataset to automatically train and tune machine learning algorithms at various hyperparameter tuning and feature selection for an optimal selection of a model that best fits the training dataset using a given target metric. 

Below is the configuration specified for automl to set and run.

automl_config = AutoMLConfig(
    task='classification',
    iterations=30,
    label_column_name='y',
    iteration_timeout_minutes=5,
    primary_metric='accuracy',
    training_data=ds,
    n_cross_validations=2)

## Pipeline comparison

From the experimental results of the Azure Machine Learning pipelines (Scikit-Learn and AutoML): the AutoML pipeline is observed to produce the best performing model, Voting Ensemble, which showed to be a best fit to the data by its 91.5% accuracy. Though, the accuracy achieved with Voting Ensemble is a marginal difference in comparsion to the 91.18% accuracy achieved with Scikit-Learn hyperparameter tuned Logistic Regression model, the performance difference can be attributed to the weighing mechanism that AutoML automatically applies to imbalanced data.


## Future work
The most common issue with any model is dealing with Bias. It highly impacts the model performance and optimization.

There are some performance improvement strategies that can be explored.

On visualizing the data by class using the Azure ML feature importance.The ‘yes’ class have more data than the ‘no’ class. Class balancing techniques can help to prevent the Scikit-Learn Logistic Regression Model from overfitting.

For the AutoML, the cross-validation hyperparameter can be experimented to find the best cross-validation fold for the given data.

Metrics such as AUC metric or the F1 metric can be optimized because they are insensitive to class imbalance.


## References
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)

