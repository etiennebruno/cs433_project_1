import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from IPython.core.display import display, HTML



############################################################################## CONSTANTS DEFINITIONS
PRI_JET_NUM_IDX = 22   
PRI_JET_NUM_VALUES = range(4)
NUMBER_GROUPS = len(PRI_JET_NUM_VALUES)
NBR_FEATURES = 30
UNDEFINED_VALUE = -999.
CALL_RIDGE = True
CALL_REG_LOGISTIC = False



############################################################################## TRAINING DATA : LOADING AND FEATURE ENGINEERING
#loading the training data
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#seperating the data within the four groups (with respect to the jet_number)
jet_groups_indices = [tX[:, PRI_JET_NUM_IDX] == pri_jet_num_value for pri_jet_num_value in PRI_JET_NUM_VALUES]
TX_arr = [tX[group_indices] for group_indices in jet_groups_indices]
Y_arr, TX_arr = zip(*[(y[group_indices], tX[group_indices]) for group_indices in jet_groups_indices])
Y_arr, TX_arr = list(Y_arr), list(TX_arr)

#collecting the indices of the undefined features for each group
undefined_features = [[], [], [], []]
for group_idx in range(NUMBER_GROUPS):
    tx = TX_arr[group_idx]
    for feature_idx in range(NBR_FEATURES):
        feature_column = tx[:, feature_idx]
        if np.all(feature_column == UNDEFINED_VALUE):
            undefined_features[group_idx].append(feature_idx)

#computing the std of the features for each group
STDS = [np.std(TX_arr[i], axis = 0) for i in range(NUMBER_GROUPS)]

#collecting the indices of the features with no variance (i.e. constant features) within each groups
cst_features = [[], [], [], []]
for group_idx, elem in enumerate(STDS):
    for feature_idx, std in enumerate(elem):
        if std == 0. and feature_idx not in undefined_features[group_idx]:
            cst_features[group_idx].append(feature_idx)

#deleting the features either undefined or with no variance (i.e. constant features) within each groups
features_to_keep = ([[x for x in range(NBR_FEATURES) 
                      if x not in undefined_features[group_idx] and x not in cst_features[group_idx]] 
                      for group_idx in range(NUMBER_GROUPS)])
TX_arr = [TX_arr[group_idx][:, features_to_keep[group_idx]] for group_idx in range(NUMBER_GROUPS)]

#computing the median of each feature and substituting it instead of undefined values
train_medians = []
for group_idx in range(NUMBER_GROUPS):
    medians = np.apply_along_axis(lambda v: np.median(v[v!=UNDEFINED_VALUE]), 0, TX_arr[group_idx])
    train_medians.append(medians)
    for col_num in range(TX_arr[group_idx].shape[1]):
        column = TX_arr[group_idx][:, col_num]
        column[column == UNDEFINED_VALUE] = medians[col_num]

#standardizing the data
#TX_arr = [standardize(TX_arr[idx]) for idx in range(NUMBER_GROUPS)]

if CALL_RIDGE:
    #applying a logarithmic transformation to the data
    for group_idx in range(NUMBER_GROUPS):
        for idx_line in range(TX_arr[group_idx].shape[0]):
            for idx_col in range(TX_arr[group_idx].shape[1]):
                if TX_arr[group_idx][idx_line][idx_col] == 0:
                    TX_arr[group_idx][idx_line][idx_col] = np.log(1e-6)         
                elif TX_arr[group_idx][idx_line][idx_col] < 0:
                    TX_arr[group_idx][idx_line][idx_col] = - np.log(-TX_arr[group_idx][idx_line][idx_col])
                else: 
                    TX_arr[group_idx][idx_line][idx_col] = np.log(TX_arr[group_idx][idx_line][idx_col])


                
############################################################################## TEST DATA : LOADING AND FEATURE ENGINEERING
#loading the test data 
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#seperating the data within the four groups (with respect to the jet_number)
jet_groups_indices_test = [tX_test[:, PRI_JET_NUM_IDX] == pri_jet_num_value for pri_jet_num_value in PRI_JET_NUM_VALUES]
TX_test_arr = list([tX_test[group_indices] for group_indices in jet_groups_indices_test])

#removing unused features (using the indices found during the processing of the training data)
TX_test_arr = [TX_test_arr[group_idx][:, features_to_keep[group_idx]] for group_idx in range(NUMBER_GROUPS)]

#replacing the the undefined values by the median of the corresponding feature
for group_idx in range(NUMBER_GROUPS):
    for col_num in range(TX_test_arr[group_idx].shape[1]):
        column = TX_test_arr[group_idx][:, col_num]
        column[column == UNDEFINED_VALUE] = train_medians[group_idx][col_num]

#standardizing the data
#TX_test_arr = [standardize(TX_test_arr[idx]) for idx in range(NUMBER_GROUPS)]

if CALL_RIDGE:
#applying a logarithmic transformation to the data
    for group_idx in range(NUMBER_GROUPS):
        for idx_line in range(TX_test_arr[group_idx].shape[0]):
            for idx_col in range(TX_test_arr[group_idx].shape[1]):
                if TX_test_arr[group_idx][idx_line][idx_col] == 0:
                    TX_test_arr[group_idx][idx_line][idx_col] = np.log(1e-6)         
                elif TX_test_arr[group_idx][idx_line][idx_col] < 0:
                    TX_test_arr[group_idx][idx_line][idx_col] = - np.log(-TX_test_arr[group_idx][idx_line][idx_col])
                else: 
                    TX_test_arr[group_idx][idx_line][idx_col] = np.log(TX_test_arr[group_idx][idx_line][idx_col])

                

#RUNNING RIDGE-REGRESSION                
if CALL_RIDGE:                  
    ###################################################################### BEST PARAMETERS SELECTION : CROSS-VALIDATION
    seed = 15
    degrees = range(1, 5)
    k_fold = 4
    lambdas = list(np.logspace(-7, 2, 25)) 
    PARAM_arr = []
    
    for group_idx in range(NUMBER_GROUPS):
        degree, lambda_ = cross_validation_demo_ridge(Y_arr[group_idx], TX_arr[group_idx], seed, degrees, k_fold, lambdas)
        PARAM_arr.append((degree, lambda_))
        print(f" ---> for group {group_idx}, the obtained best degree is {degree} and lambda is {lambda_}")
            
    ###################################################################### TRAINING AND GENERATING THE PREDICTIONS
    #training model and generating the predictions for each group
    y_pred = np.empty(tX_test.shape[0])
    for group_idx in range(NUMBER_GROUPS):
        #training
        tx_train = build_poly(TX_arr[group_idx], PARAM_arr[group_idx][0])
        y_train = Y_arr[group_idx]
        lambda_ = PARAM_arr[group_idx][1]
        weight, loss = ridge_regression(y_train, tx_train, lambda_)
        
        #prediction
        tx_test = build_poly(TX_test_arr[group_idx], PARAM_arr[group_idx][0])
        y_pred[jet_groups_indices_test[group_idx]] = predict_labels(weight, tx_test).flatten()
    
    #creating csv file
    OUTPUT_PATH = '../data/sample-submission_ridge.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

#RUNNING LOGISTIC-REGRESSION                
if CALL_REG_LOGISTIC:
    ###################################################################### BEST PARAMETERS SELECTION : CROSS-VALIDATION    
    seed = 15
    degrees= range(7)
    k_fold = 4
    max_iters = 2_000
    lambdas = np.logspace(-7, 2, 10)
    gammas = np.logspace(-6, -2, 10)   
    PARAM_arr = []
    
    for group_idx in range(NUMBER_GROUPS):
        y=np.array(Y_arr[group_idx])
        y[y == - 1.0] = 0.0
        tX=np.array(TX_arr[group_idx])
        initial_w = np.zeros(len(features_to_keep[group_idx]))
        degree, gamma, lambda_ = cross_validation_demo_reg_logistic(y, tX, max_iters, seed, degrees, k_fold, lambdas, gammas)
        PARAM_arr.append((degree, lambda_))
        print(f" ---> for group {group_idx}, the obtained best degree is {degree} and lambda is {lambda_}")
            
    ###################################################################### TRAINING AND GENERATING THE PREDICTIONS
    #training model and generating the predictions for each group
    y_pred = np.empty(tX_test.shape[0])
    for group_idx in range(NUMBER_GROUPS):
        #training
        max_iters = 5_000
        degree = PARAM_arr[group_idx][0]
        gamma = PARAM_arr[group_idx][1]
        lambda_ = PARAM_arr[group_idx][2]
        tx_train = build_poly(TX_arr[group_idx], degree)
        y_train = Y_arr[group_idx]
        y_train[y_train == -1.0] = 0.0
        initial_w = np.zeros((tx_train.shape[1], 1))  
        weight, loss = reg_logistic_regression(y_train, tx_train, lambda_, initial_w, max_iters, gamma)
        
        #prediction
        tx_test = build_poly(TX_test_arr[group_idx], PARAM_arr[group_idx][0])
        y_pred[jet_groups_indices_test[group_idx]] = predict_labels(weight, tx_test).flatten()
    
    #creating csv file
    OUTPUT_PATH = '../data/sample-submission_regularized_logistic.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)