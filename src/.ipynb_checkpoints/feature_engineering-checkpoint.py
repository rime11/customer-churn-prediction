import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_target(data):
    '''
    input: cleaned data
    output: maps target values with 1 and 0 and returns the mapped target
    '''
    y = data['Churn'].map({'Yes':1,'No':0})
    return y
    
def scale_encode_data(data, non_num_cols, is_train = True, scaler=None):
    '''
    Transforms data by scaling numeric features and encoding non_numeric features
    
    Parameters: cleaned_data
    Returns: scaled data with customerID and Churn dropped
    '''
    #copy dataframe
    
    data = data.copy()
    
    #drop churn and customerID
    data = data.drop(['customerID','Churn'], axis=1)
    
    #map binary features 
    data['gender'] = data['gender'].map({'Male':1,'Female':0})
    data['Partner'] = data['Partner'].map({'Yes':1,'No':0})
    data['Dependents'] = data['Dependents'].map({'Yes':1,'No':0})
    data['PhoneService'] = data['PhoneService'].map({'Yes':1,'No':0})
    data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes':1,'No':0})

    #encode other non numeric features
    non_numeric_dummies =pd.get_dummies(data[non_num_cols], drop_first=True)#get_dummies gives  binary features
    #change values to numeric 
    dummies_num = non_numeric_dummies.astype(int)

    #concat to original dataframe and drop non-numeric original columns
    df_num = pd.concat([data.drop(non_num_cols,axis=1),dummies_num], axis=1)
    df_num= df_num.reset_index(drop = True)
    
    #scale 
    if is_train:
        #Scale all features
        scaler=StandardScaler()
        scaled_df = pd.DataFrame(
            scaler.fit_transform(df_num),columns=df_num.columns)     
        
    else:
     # For test data, use the pre-fitted scaler
        if scaler is None:
            raise ValueError("Scaler must be provided for test data")
        else:
            scaled_df = pd.DataFrame(scaler.transform(df_num),columns=df_num.columns)
    
    return scaled_df, scaler

