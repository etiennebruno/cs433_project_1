# Machine Learning Project 1 - Higgs Boson - 2021-2022

The purpose of this project is to implement machine learning algorithms for binary classification of the Higgs boson dataset. The resulting predictions are submitted to [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

## Team Members
Sami FERCHIOU : sami.ferchiou@epfl.ch <br/>
Miguel SANCHEZ : miguel-angel.sanchezndoye@epfl.ch <br/>
Etienne BRUNO : etiene.bruno@epfl.ch <br/>

##  Aim
The aim of this project is to find a model able to predict whether a given set of measurements (original data from the CERN) represents the emission of a Higgs boson particle or can be considered as background noise (i.e. is due to physical phenomena we are not interested in). In this project, we implement six different machine learning algorithms on data whose features are carefully chosen, cleaned and eventually improved. Using grid-search and cross-validation, we are able to optimize the hyper parameters. With an accuracy of 80.6\%, ridge regression was found to be the best performing algorithm for this problem.

## Requirements
Please make sure you have Python 3 with the following packages installed :
- numpy
- matplotlib
- seaborn

## Instructions
Clone the repository using the following command via ssh:
```
git clone git@github.com:etiennebruno/ml_project_1.git
```
Please run the following commands on a terminal to run the main program
```
cd script
python run.py
```

## Overview
Here's a list of the relevant source files 

|Source file | Description|
|---|---|
| `implementations.py`  | Regrouping the six machine learning algorithms we hqve developped for this project as well as dependant function|
|`run.py`               | Main script containing the solution of the problem producing our highest prediction score|
|`helpers.py`           | Containing additional functions used in the project|
|`projet1.ipynb`        | Notebook of the project with all the visualization and the analysis of the training data as weel as the code of the training models|
