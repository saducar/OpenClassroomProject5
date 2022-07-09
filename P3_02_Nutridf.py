

# import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import KNNImputer
#import missingno as msno
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")

from scipy.stats import f_oneway

import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

    
def sugars_100g_N(x):
    if np.isnan(x):
        return np.nan
    elif (x <= 4.5):
        return 0
    elif ( x<=9):
        return 1
    elif (x<=13.5):
        return 2
    elif (x<=18):
        return 3
    elif (x<=22.5):
        return 4
    elif (x<=27):
        return 5
    elif (x<=31):
        return 6
    elif (x<=36):
        return 7
    elif (x<=40):
        return 8
    elif (x<=45):
        return 9
    else:
        return 10
 

    
def energy_100g_N(x):
    if(np.isnan(x)):
        return np.nan 
    elif(x <= 335):
        return 0
    elif(x > 335) & (x <= 670):
        return 1
    elif(x > 670) & (x <= 1005):
        return 2
    elif(x > 1005) & (x <= 1340):
        return 3
    elif(x > 1340) & (x <= 1675):
        return 4
    elif(x > 1675) & (x <= 2010):
        return 5
    elif(x > 2010) & (x <= 2345):
        return 6
    elif(x > 2345) & (x <= 2680):
        return 7
    elif(x > 2680) & (x <= 3015):
        return 8
    elif(x > 3015) & (x <= 3350):
        return 9
    else:
        return 10
    
def saturated_fat_100g_N(x):
    if(np.isnan(x)):
        return np.nan
    elif(x <= 1):
        return 0
    elif(x > 1) & (x <= 2):
        return 1
    elif(x > 2) & (x <= 3):
        return 2
    elif(x > 3) & (x <= 4):
        return 3
    elif(x > 4) & (x <= 5):
        return 4
    elif(x > 5) & (x <= 6):
        return 5
    elif(x > 6) & (x <= 7):
        return 6
    elif(x > 7) & (x <= 8):
        return 7
    elif(x > 8) & (x <= 9):
        return 8
    elif(x > 9) & (x <= 10):
        return 9
    else:
        return 10


def sodium_100g_N(x):
    if(np.isnan(x)):
        return np.nan
    elif(x <= 0.090):
        return 0 
    elif(x > 0.090 and x <= 0.180):
        return 1
    elif(x > 0.180 and x <= 0.270):
        return 2
    elif(x > 0.270 and x <= 0.360):
        return 3
    elif(x > 0.360 and x <= 0.450):
        return 4
    elif(x > 0.450 and x <= 0.540):
        return 5
    elif(x > 0.540 and x <= 0.630):
        return 6
    elif(x > 0.630 and x <=0.720):
        return 7
    elif(x > 0.720 and x <= 0.810):
        return 8
    elif(x > 0.810 and x <= 0.900):
        return 9
    else:
        return 10


def proteins_100g_N(x):
    if np.isnan(x):
        return np.nan
    elif (x >0 and x<=0.9):
        return 0
    elif (x > 0.9 and x<=1.6):
        return 1
    elif (x > 1.6 and x<=3.2):
        return 2
    elif (x >3.2 and x<=4.8):
        return 3
    elif (x >4.8 and x<=6.4):
        return 4
    else:
        return 5
    
def nutriscore_grade(x):
    if np.isnan(x):
        return np.nan
    elif (x <= -1):
        return 'a'
    elif (x >= 0 and x <= 2):
        return 'b'
    elif (x >= 3 and x <= 10):
        return 'c'
    elif (x >= 11 and x <= 18):
        return 'd'
    #if (x >= 19 and <= 40):
    else:
        return 'e'
    
def nutri_score(energy,sugars,sodium,saturated_fat,proteins):
    if(np.isnan(energy) or np.isnan(sugars) or np.isnan(sodium) or np.isnan(saturated_fat) or np.isnan(proteins)):
        return np.nan
    else:
        return (energy + sugars + sodium + saturated_fat - proteins -3)
    
def NutriClean(data):
    '''
    this fuction  is to clean nutrigrade 
    '''
    clean_data = pd.DataFrame([])
    Data = pd.DataFrame([])
    # Removing all those columns which are not necessary
    Data = data # add any columns you need 
    
     # changing Values of various columns inbte 0 - 10 
    Data['sugars_100g_N'] = Data['sugars_100g'].apply( sugars_100g_N)
    Data['energy_100g_N'] = Data['energy_100g'].apply(lambda x : energy_100g_N(x))
    Data['saturated-fat_100g_N'] = Data['saturated-fat_100g'].apply(lambda x : saturated_fat_100g_N(x))
    Data['sodium_100g_N'] = Data['sodium_100g'].apply(lambda x : sodium_100g_N(x))
    Data['proteins_100g_N'] = Data['proteins_100g'].apply(lambda x : proteins_100g_N(x))
    
        #getting nutri score
    Data['nutriscore_score'] = Data.apply(lambda row :nutri_score(row['energy_100g_N'] , row['sugars_100g_N'] ,row['sodium_100g_N'] , row['saturated-fat_100g_N'] , row['proteins_100g_N']),axis = 1)
    # getting nutrigrade
    Data['nutriscore_grade'] = Data['nutriscore_score'].apply(lambda x : nutriscore_grade(x))

    #Remove unwanted columns like the virtula energy, total gram and _N¶
    clean_data = Data#[[ "nova_group",'nutriscore_score' ,'nutriscore_grade','energy_100g',"energy-kcal_100g",'fat_100g',
                      # 'saturated-fat_100g','carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
                       #'salt_100g','sodium_100g',"countries_en"]]
    
    return clean_data