from scipy.stats import boxcox
import numpy as np

def Gender_encoding(gender):
    
    if gender == 'Female':
        return 0
    else:
        return 1
    
def Family_history_with_overweight_encoding(family_history_with_overweight):
    
    if family_history_with_overweight == 'no':
        return 0
    else:
        return 1
    
def FAVC_encoding(favc):
    
    if favc == 'no':
        return 0
    else:
        return 1
    
def CAEC_encoding(caec):
    
    if caec == 'Sometimes':
        return 0
    elif caec == 'Frequently':
        return 1
    elif caec == 'Always':
        return 2
    elif caec == 'no':
        return 3
    
def SMOKE_encoding(smoke):
    
    if smoke == 'no':
        return 0
    else:
        return 1
    
def SCC_encoding(scc):
    
    if scc == 'no':
        return 0
    else:
        return 1
    
def CALC_encoding(calc):
    
    if calc == 'no':
        return 0
    elif calc == 'Sometimes':
        return 1
    elif calc == 'Frequently':
        return 2
    elif calc == 'Always':
        return 3
    
def MTRANS_encoding(trans):
    
    if trans == 'Public_Transportation':
        return 0
    elif trans == 'Walking':
        return 1
    elif trans == 'Automobile':
        return 2
    elif trans == 'Motorbike':
        return 3
    elif trans == 'Bike':
        return 4
    
def Age_Transformation(ag):
    
    boxcox_lambda_val_age = -1.5458226415652254
    New = ((ag**boxcox_lambda_val_age) - 1) / boxcox_lambda_val_age
    
    return New

def NCP_Transformation(ncp):
    
    boxcox_lambda_val_NCP = 2.223276247845529
    New = ((ncp**boxcox_lambda_val_NCP) - 1) / boxcox_lambda_val_NCP
   
    return New

def converter(pred):
    
    if pred == 0:
        return 'Normal_Weight'
    elif pred == 1:
        return 'Overweight_Level_I'
    elif pred == 2:
        return 'Overweight_Level_II'
    elif pred == 3:
        return 'Obesity_Type_I'
    elif pred == 4:
        return 'Insufficient_Weight'
    elif pred == 5:
        return 'Obesity_Type_II'
    elif pred == 6:
        return 'Obesity_Type_III'