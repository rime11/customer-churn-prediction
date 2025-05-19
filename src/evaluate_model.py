import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from src.feature_engineering import extract_target, scale_encode_data
from src.custom_transformer import CustomFeatureProcessor

def evaluate_model(data, processing_func,non_num_cols, is_train = True, scaler=None):
    '''
    Evaluate RandomForest using a custom processing function

    parameters: 
        pandas df: cleaned IBM churn data
        feature processing function that takes the data and returns scaled, encoded data
    Returns:
        dict: cross-validation results
    '''
    #target variable
    target = extract_target(data)
    
    
    #create a pipeline with feature processing
    pipeline = Pipeline([
                    ('processor',CustomFeatureProcessor(processing_func,non_num_cols=non_num_cols, is_train = True, scaler=None)),
                    ('classifier',RandomForestClassifier(class_weight='balanced',random_state=42))
                        ])
    
    #calculate feature importances
    pipeline.fit(data, target)
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = (pipeline.named_steps['processor'].transform(data)).columns

    #stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
            'precision':make_scorer(precision_score),
            'recall':make_scorer(recall_score),
            'f1_score':make_scorer(f1_score),
            'auc_score':make_scorer(roc_auc_score)
          } 
    
    #cross validation
    cv_results = cross_validate(pipeline, data, target, cv=skf, scoring=scoring)

    
    return cv_results, importances, feature_names
    

def predict_on_test_data(train_data, test_data, processing_func, non_num_cols):
    '''
    predict using test set 
    ___________
    Parameters:
        training data, test data, the processing fucntion used to encode the data, the non numeric columns
    ___________
    Returns:
        predictions
        probabilities
    '''
    # Extract target from training data
    y_train = extract_target(train_data)
    y_test = extract_target(test_data)
    
    # Create pipeline
    pipeline = Pipeline([
        ('processor', CustomFeatureProcessor(processing_func, non_num_cols=non_num_cols, is_train=True)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Fit pipeline on training data
    pipeline.fit(train_data, y_train)

    # Make predictions on test data
    predictions = pipeline.predict(test_data)
    probabilities = pipeline.predict_proba(test_data)[:, 1]
    
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc_roc = roc_auc_score(y_test, predictions)
    
    return predictions, precision, recall, f1, auc_roc

    

def evaluate_model_reduced_features(data, processing_func,non_num_cols, is_train = True, scaler=None):
    '''
    Evaluate RandomForest using a custom processing function
    Add a selector that will only select features that have importance higher than the half the mean of importances

    parameters: 
        pandas df: cleaned IBM churn data
        feature processing function that takes the data and returns scaled, encoded data
    Returns:
        dict: cross-validation results, importances, and feature names
    '''
    #target variable
    target = extract_target(data)
    
    #create a pipeline with feature processing
    
    pipeline = Pipeline([
                    ('processor',CustomFeatureProcessor(processing_func,non_num_cols=non_num_cols, is_train = True, scaler=None)),
                    ('classifier',RandomForestClassifier(class_weight='balanced',random_state=42))
                        ])

    
    #calculate feature importances
    pipeline.fit(data, target)
    
    #add a selector that will keep only features that have feature importance above mean importances
    selector = SelectFromModel(pipeline.named_steps['classifier'], threshold='0.5*mean') 
    pipeline_reduced = Pipeline([
                        ('processor',CustomFeatureProcessor(processing_func,non_num_cols=non_num_cols, is_train = True, scaler=None)),
                        ('selector',selector),
                        ('classifier',RandomForestClassifier(class_weight='balanced',random_state=42))
                        ])
    pipeline_reduced.fit(data, target)
    
    importances = pipeline_reduced.named_steps['classifier'].feature_importances_
    feature_names = (pipeline_reduced.named_steps['processor'].transform(data)).columns
    feature_index = pipeline_reduced.named_steps['selector'].get_support()
    features = feature_names[feature_index]
    

    #stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {'precision':make_scorer(precision_score),
                'recall':make_scorer(recall_score),
                'f1_score':make_scorer(f1_score),
              'auc_score':make_scorer(roc_auc_score)
              } 
    
    #cross validation
    cv_results = cross_validate(pipeline_reduced, data, target, cv=skf, scoring=scoring)

    
    return cv_results, importances, features
    

def predict_on_test_data_reduced_features(train_data, test_data, processing_func, non_num_cols):
    '''
    predict using test set 
    ___________
    Parameters:
        training data, test data, the processing fucntion used to encode the data, the non numeric columns
    ___________
    Returns:
        predictions
        probabilities
    '''
    # Extract target from training data
    y_train = extract_target(train_data)
    y_test = extract_target(test_data)
    
    # Create pipeline
    pipeline = Pipeline([
        ('processor', CustomFeatureProcessor(processing_func, non_num_cols=non_num_cols, is_train=True)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Fit pipeline on training data
    pipeline.fit(train_data, y_train)

    selector = SelectFromModel(pipeline.named_steps['classifier'], threshold='0.5*mean') 
    pipeline_reduced = Pipeline([
                        ('processor',CustomFeatureProcessor(processing_func,non_num_cols=non_num_cols, is_train = True, scaler=None)),
                        ('selector',selector),
                        ('classifier',RandomForestClassifier(class_weight='balanced',random_state=42))
                        ])
    
    pipeline_reduced.fit(train_data, y_train)
    #Get new feature indeces
  
    # Make predictions on test data
    predictions = pipeline_reduced.predict(test_data)
    #probabilities = pipeline.predict_proba(test_data)[:, 1]
    
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc_roc = roc_auc_score(y_test, predictions)
    
    return predictions, precision, recall, f1, auc_roc