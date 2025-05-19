import pandas as pd
import numpy as np

def commitment_score(row):
    '''
    Asses commitment based on contract and payment method
    
    parameters: a row in the dataframe
    return: commitment score [high, med_high, med_low, low]
    '''
    
    if row['Contract'] == 'Two year':
        if row['PaymentMethod'] in ['Bank transfer (automatic)','Credit card (automatic)']:
            return 'high'
        else:
            return 'med_high'
    elif row['Contract'] == 'One year':
        if row['PaymentMethod'] in ['Bank transfer (automatic)','Credit card (automatic)']:
            return 'med_high'
        else:
            return 'med_low'
    else:
        if row['PaymentMethod'] in ['Bank transfer (automatic)','Credit card (automatic)']:
            return 'med_low'
        else:
            return 'low'

def risk_score_calculator(row):
    '''
    calculates the risk score according to customers' payment method
    '''
    risk_score = 0
    #Contract
    if row['Contract'] == 'Month-to-month':
        risk_score = 3
    elif row['Contract'] == 'One year':
        risk_score = 2
    #if two_year then 1
    else:
        risk_score = 1
    #PaymentMethod
    if row['PaymentMethod'] == 'Electronic check':
        risk_score += 4
    elif row['PaymentMethod'] == 'Mailed check':
        risk_score += 3
    elif row['PaymentMethod'] == 'Credit card (automatic)':
        risk_score += 2
    #if Bank transfer (automatic) then 1
    else:
        risk_score += 1
    #PaperlessBilling
    if row['PaperlessBilling'] == 'Yes':
        risk_score +=1
    # if PaperlessBilling = no then 0
    return risk_score

def family_structure(row):
    '''returns family structure for partner and dependents columns
    '''
    if row['Partner'] == 'Yes':
        if row['Dependents'] == 'Yes':
            status = 'family'
        else: # has a partner but no dependents
            status = 'couple'
    else: # has no partner
        if row['Dependents'] == 'Yes':
            status = 'single_parent'
        else: # not partner and no dependents
            status = 'single'
    return status

def add_features(data):
    
    data = data.copy()
    # add discount
    data['discount']= np.where(data['MonthlyCharges']*data['tenure'] > data['TotalCharges'],1,0)
    
    #How many security services does a customer have: OnlineBackup, OnlineSecurity, TechSupport, DeviceProtection
    security_serv = data[['OnlineBackup','OnlineSecurity','TechSupport','DeviceProtection']]
    data['service_sum'] = security_serv.map(lambda x: 1 if x=='Yes' else 0).sum(axis=1)

    #commitment_level
    data['commitment_level'] = data.apply(commitment_score, axis=1)
    
    #payment_risk_score
    data['payment_risk_score'] = data.apply(risk_score_calculator, axis=1)
    
    #family_status
    data['family_status'] = data[['Partner','Dependents']].apply(family_structure, axis=1)
    
    '''
    Entertainment engagement: Measure entertainemtn engagement using rows: StreamingTV and StreamingMovies which are perfectly correlated. 
    If customers have both then the score is 2, if only 1 then score is 1 if none or no internet service then 0
    '''
    
    data['engagement_score']= data[['StreamingTV','StreamingMovies']].map(lambda x: 1 if x=='Yes' else 0).sum(axis=1)

    # Customers with premium service: they would have PhoneService and MultipleLines and fiber optics for InternetService
    data['premium_service'] = ((data['PhoneService'] == 'Yes')& 
                                 (data['MultipleLines'] == 'Yes')& 
                                 (data['InternetService'] == 'Fiber optic')).astype('int')
    # Tenure bins (Q1, Q2, Q3, Q4): bin tenure into 4 bins and label them quarter 1 to quarter 4
    labels = ['Q1','Q2','Q3','Q4']
    data['tenureQuartile'] = pd.cut(data['tenure'],bins = 4,labels=labels)
    
    #Customers that are at high churn risk are in Q1
    data['churn_high_risk'] = (data['tenureQuartile']=='Q1').astype(int)

    #create labels for monthlyCharges and totalCharges
    labels = ['low','medium','high']
    data['monthlyChargesBins'] = pd.cut(data['MonthlyCharges'],bins=3, labels=labels)
    data['totalChargesBins'] = pd.cut(data['TotalCharges'],bins=3, labels=labels)

    #Any customer who spends higher than 75% of all customer will be considered a premium customer
    data['IsPremium'] = (data['MonthlyCharges']>data['MonthlyCharges'].quantile(.75)).astype(int)

    #tenure squared 
    data['tenureSquared'] = data['tenure']**2

    #average monthly lifetime spend
    data['historicalAvgSpending'] = data.apply(lambda x: x['MonthlyCharges'] if x['tenure'] == 0 
                                                       else x['TotalCharges']/x['tenure'], axis=1)
   #create labels for monthlyCharges and totalCharges
    labels = ['low','medium','high']
    data['monthlyChargesBins'] = pd.cut(data['MonthlyCharges'],bins=3, labels=labels)
    data['totalChargesBins'] = pd.cut(data['TotalCharges'],bins=3, labels=labels)

    #a premium customer is one who spends higher than 75% of all customer 
    data['IsPremium'] = (data['MonthlyCharges']>data['MonthlyCharges'].quantile(.75)).astype(int)

    #Average monthly lifetime spend = monthlyCharges / tenure
    data['historicalAvgSpending'] = data.apply(lambda x: x['MonthlyCharges'] if x['tenure'] == 0 
                                                       else x['TotalCharges']/x['tenure'], axis=1)
    #spendingRatio
    data['spendingRatio'] = data['MonthlyCharges'] / data['historicalAvgSpending']

    # customer's value: how much they spend over their lifetime ==> monthlyCharges*tenure
    labels = ['low','medium','high','top']
    customerValue = data['MonthlyCharges']*data['tenure']
    tiers = pd.cut(customerValue, bins = 4, labels=labels)
    data['valueTier'] = tiers

    # Projected customer value: monthlyCharges * 12 should return the projected customer value after tiers will be created
    data['projectedValue'] = data['MonthlyCharges']* 12
    #create bins
    labels = ['low','medium','high','top']
    data['projectedValueTier'] = pd.cut(data['projectedValue'],bins=4, labels=labels)

    # Spending intensity: it accounts for both spending and loyalty, higher values mean customers spend more relative to their tenure
    data['spendingIntensity'] = data['projectedValue']/(data['tenure']+1)
    labels = ['low','medium','high']
    tiers = pd.cut(data['spendingIntensity'], bins = 3, labels=labels)
    data['spendingIntensityTier'] = tiers

    return data


