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

def load_data(filename):
    df = pd.read_csv(filename)
    return df


def fill_energy(carbohydrates,proteins,fat):
    if (np.isnan(carbohydrates) or np.isnan(proteins) or np.isnan(fat)):
        return np.nan
    else:
        energy = (carbohydrates*17) + (proteins * 17) + (fat * 37)
        return energy

def fill_protein(energy,carbohydrates,fat):
    if (np.isnan(energy) or np.isnan(carbohydrates) or np.isnan(fat)):
        return np.nan
    else:
        protein = (energy - (fat * 37) - (carbohydrates*17)) / 17
        return protein


def fill_fat(energy,proteins,carbohydrates):
    if (np.isnan(energy) or np.isnan(proteins) or np.isnan(carbohydrates)):
        return np.nan
    else:
        fat = (energy - (proteins * 17) - (carbohydrates*17)) / 37
        return fat 

def fill_salt(sodium):
    if (np.isnan(sodium)):
        return np.nan
    else:
        salt = sodium/ 2.5  
        return salt

def fill_sodium(salt):
    if (np.isnan(salt)):
        return np.nan
    else:
        sodium = salt*2.5
        return sodium 

def fill_energy_kcal(energy):
    if (np.isnan(energy)):
        return np.nan
    else:
        kcal = (energy)/ 4.184  
        return kcal

def fill_sugar(energy,proteins,carbohydrates,fat):
    if (np.isnan(energy) or np.isnan(proteins) or np.isnan(carbohydrates) or np.isnan(fat)):
        return np.nan
    else:
        sugars = (energy- proteins*17  - carbohydrates*17 - fat*37) /3.87
        return sugars
    
def fill_potassium(fat,proteins,carbohydrates,sodium,cholesterol):
     if (np.isnan(fat) or np.isnan(proteins) or np.isnan(carbohydrates) or np.isnan(sodium),np.isnan(cholesterol)):
        return np.nan
     else:
        pot = 100- fat - proteins - carbohydrates- sodium- cholesterol
        return pot
    
    
    
def proteins_100g_virtual(fat,potassium,carbohydrates,sodium,cholesterol):
    if (np.isnan(fat) or np.isnan(potassium) or np.isnan(carbohydrates) or np.isnan(sodium),np.isnan(cholesterol)):
        return np.nan
    else:
        pro = 100- fat - potassium - carbohydrates- sodium- cholesterol
        return pro
    
def virtual_energy(proteins,carbohydrates,fat):
    if (np.isnan(carbohydrates) or np.isnan(proteins) or np.isnan(fat)):
        return np.nan
    else:
        energy = (carbohydrates*17) + (proteins * 17) + (fat * 37)
        return energy
    
    
#  if row is bigger by 10% it will replace the amount with the Virtual data
def replace_energy(energy,virtual):
    if (np.isnan(energy) or  np.isnan(virtual)):
        return np.nan
    elif(energy < (virtual*0.9)):
        return virtual
    else:
        return energy
    
def sugars_100g_N(x):
    if np.isnan(x):
        return np.nan
    elif (x <= 4.5):
        return 0
    elif (x > 4.5 and x<=9):
        return 1
    elif (x > 9 and x<=13.5):
        return 2
    elif (x >13.5 and x<=18):
        return 3
    elif (x >18 and x<=22.5):
        return 4
    elif (x >22.5 and x<=27):
        return 5
    elif (x >27 and x<=31):
        return 6
    elif (x >31 and x<=36):
        return 7
    elif (x >36 and x<=40):
        return 8
    elif (x >40 and x<=45):
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
    
def nutri_score(energy,sugars,sodium,saturated_fat,proteins):
    if(np.isnan(energy) or np.isnan(sugars) or np.isnan(sodium) or np.isnan(saturated_fat) or np.isnan(proteins)):
        return np.nan
    else:
        return (energy + sugars + sodium + saturated_fat - proteins -2)
    
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
        
        
def clean_data(data):
    
    clean_data = pd.DataFrame([])
    Data = pd.DataFrame([])
    # Removing all those columns which are not necessary
    Data = data[[ 'nutriscore_score' ,'nutriscore_grade','energy-kcal_100g', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
                    'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 
                    'salt_100g', 'sodium_100g',"cholesterol_100g","potassium_100g","countries_en"]]
    
    # removing the the cols which have 100 % empty
    perc = 100 # Here N is 25
    min_count =  int(((100-perc)/100)*Data.shape[1] + 1)
    Data = Data.dropna( axis=0, thresh=min_count)
    
    
    
    #creating a new column to see the total weight of the product and remove the above weight
    
    
    Data["Total_grams"]=Data.apply(lambda row: row.proteins_100g + row.carbohydrates_100g+ row.fat_100g, axis=1)
    Data.drop(Data[Data['Total_grams'] > 100].index, inplace = True)
    Data.drop(Data[Data['Total_grams'] <= 0].index, inplace = True)
    
    #fill empty colum with formula's
    #filling energy_100g
    Data['energy_100g'] = Data.apply(lambda row : fill_energy(row['carbohydrates_100g'],row['proteins_100g'],row['fat_100g']) ,axis=1)
    #filling proteins_100g
    Data['proteins_100g'] = Data.apply(lambda row: fill_protein(row['energy_100g'], row['carbohydrates_100g'],row['fat_100g'] ), axis=1)
    #filling fat_100g
    Data['fat_100g'] = Data.apply(lambda row: fill_fat(row['energy_100g'], row['proteins_100g'],row['carbohydrates_100g']), axis=1)
    #filling salt_100g
    Data['salt_100g'] = Data.apply(lambda row: fill_salt(row['sodium_100g']),axis=1)
    #filling sodium_100g                             
    Data['sodium_100g'] = Data.apply(lambda row: fill_sodium(row['salt_100g']),axis=1)                                 
    #filling energy_kcal_100g
    Data['energy-kcal_100g'] = Data.apply(lambda row: fill_energy_kcal(row['energy_100g']),axis=1)
    #filling sugars_100g
    Data['sugars_100g'] = Data.apply(lambda row:fill_sugar(row['energy_100g'],row['proteins_100g'],
                                                           row['carbohydrates_100g'],row['fat_100g']) , axis=1)
    
    Data['potassium_100g'] = Data.apply(lambda row: fill_potassium(row['fat_100g'],row['proteins_100g'],row['carbohydrates_100g'],
                                                                           row['sodium_100g'],row['cholesterol_100g']),axis=1)
    
    Data["Total_grams"]=Data.apply(lambda row: row.proteins_100g + row.carbohydrates_100g+ row.fat_100g, axis=1)
    
    Data['proteins_100g_virtual'] = Data.apply(lambda row: proteins_100g_virtual( row['fat_100g']   ,row['potassium_100g'],row['carbohydrates_100g'], row['sodium_100g'], row['cholesterol_100g']),axis=1)
                                                       
    Data['proteins_100g'] = Data.apply(lambda row: replace_energy(row['proteins_100g'],row['proteins_100g_virtual']),axis = 1)
    
    Data["virtual_energy"]=Data.apply(lambda row: virtual_energy(row.proteins_100g ,row.carbohydrates_100g, row.fat_100g), axis=1)
                                                       
    
    # Removing Data from the all columns that are above and below normal parameter and redox box plot
    Data.drop(Data[Data['energy-kcal_100g'] > 900].index, inplace = True)
    Data.drop(Data[Data['fat_100g'] > 100].index, inplace = True)
    Data.drop(Data[Data['saturated-fat_100g'] > 100].index, inplace = True)
    Data.drop(Data[Data['carbohydrates_100g'] > 100].index, inplace = True)
    Data.drop(Data[Data['sugars_100g'] > 100].index, inplace = True)
    Data.drop(Data[Data['proteins_100g'] > 100].index, inplace = True)
    Data.drop(Data[Data['salt_100g'] > 100].index, inplace = True)
    Data.drop(Data[Data['energy_100g'] > 3780].index, inplace = True)
    
    Data.drop(Data[Data['fat_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['carbohydrates_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['sugars_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['salt_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['saturated-fat_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['energy-kcal_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['proteins_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['energy_100g'] <= 0].index, inplace = True)
    
    Data.drop(Data[Data['Total_grams'] > 100].index, inplace = True)
    Data.drop(Data[Data['Total_grams'] <= 0].index, inplace = True)
    
    '''
    Data.drop(Data[Data['energy-kcal_100g'] > 900  or Data['energy-kcal_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['fat_100g'] > 100 or Data['fat_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['saturated-fat_100g'] > 100 or Data['saturated-fat_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['carbohydrates_100g'] > 100 or Data['carbohydrates_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['sugars_100g'] > 100 or Data['sugars_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['proteins_100g'] > 100 or Data['proteins_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['salt_100g'] > 100 or Data['salt_100g'] <= 0].index, inplace = True)
    Data.drop(Data[Data['energy_100g'] > 3780 or Data['energy_100g'] <= 0].index, inplace = True)
    '''
    
    #Creating a new colum to see if the actual energy is close to calculated energy and replace it with the correct one
    Data["virtual energy"]=Data.apply(lambda row: (row.proteins_100g)*4 + (row.carbohydrates_100g)*4+ (row.fat_100g)*9, axis=1)
    Data['energy_100g'] = Data.apply(lambda row: replace_energy(row['energy_100g'],row['virtual energy']),axis = 1)

    # changing Values of various columns inbte 0 - 10 
    Data['sugars_100g_N'] = Data['sugars_100g'].apply(lambda x : sugars_100g_N(x))
    Data['energy_100g_N'] = Data['energy_100g'].apply(lambda x : energy_100g_N(x))
    Data['saturated-fat_100g_N'] = Data['saturated-fat_100g'].apply(lambda x : saturated_fat_100g_N(x))
    Data['sodium_100g_N'] = Data['sodium_100g'].apply(lambda x : sodium_100g_N(x))
    Data['proteins_100g_N'] = Data['proteins_100g'].apply(lambda x : proteins_100g_N(x))
    
    #getting nutri score
    Data['nutriscore_score'] = Data.apply(lambda row :nutri_score(row['energy_100g_N'] , row['sugars_100g_N'] ,row['sodium_100g_N'] , row['saturated-fat_100g_N'] , row['proteins_100g_N']),axis = 1)
    # getting nutrigrade
    Data['nutriscore_grade'] = Data['nutriscore_score'].apply(lambda x : nutriscore_grade(x))
    
    
    #Remove unwanted columns like the virtula energy, total gram and _N¶
    clean_data = Data[[ 'nutriscore_score', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
                    'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'sodium_100g']]
    
    return clean_data


def imputer(data,type = 'simple',method = 'mean',k = 3):
    if (type == 'simple'):
        #simple imputer
        if (method == 'mean'):
            #filling nan data with mean values
            si = sklearn.impute.SimpleImputer(strategy='mean')
            new_data = si.fit_transform(data)
        else:
            #filling nan data with median values
            si = sklearn.impute.SimpleImputer(strategy='median')
            new_data = si.fit_transform(data)
    else:
        #knn imputer
        knn = sklearn.impute.KNNImputer(n_neighbors=k)
        new_data = knn.fit_transform(data)
        new_df = pd.DataFrame(new_data,columns = data.columns )
    return new_df


def get_bound(outlier_df, i):
    q1 = outlier_df[i].quantile(0.25)
    q3 = outlier_df[i].quantile(0.75)
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1- (1.5*iqr)
    return lower_bound,upper_bound

def handle_outlier(data,method = 'drop',check = 'None'):
    cols = data.select_dtypes(exclude = 'object').columns
    z = np.abs(stats.zscore(data[cols].dropna(axis = 0)))
    outlier_df = pd.DataFrame(z,columns = cols)

    if (check == 1 and method == 'mean'):
        for i in outlier_df.columns:
            x = 1
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].mean()

    elif (check == 2 and method == 'mean'):
        for i in outlier_df.columns:
            x = 2
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].mean()

    elif (check == 3 and method == 'mean'):
        for i in outlier_df.columns:
            x = 3
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].mean()

    elif (check == 1 and method == 'median'):
        for i in outlier_df.columns:
            x = 1
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].median()

    elif (check == 2 and method == 'median'):
        for i in outlier_df.columns:
            x = 2
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].median()

    elif (check == 3 and method == 'median'):
        for i in outlier_df.columns:
            x = 3
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].median()

    elif (check == 'None' and method == 'median'):
        for i in outlier_df.columns:
            x = get_bound(outlier_df, i)[0]
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].median()

    elif (check == 'None' and method == 'mean'):
        for i in outlier_df.columns:
            x = get_bound(outlier_df, i)[0]
            h_outlier_df = outlier_df.loc[outlier_df[i] >x,i]= outlier_df[i].median()

    elif (check == 1 and method == 'drop'):
        for i in outlier_df.columns:
            x = 1
            h_outlier_df = outlier_df.drop(outlier_df[outlier_df[i] > x].index)

    elif (check == 2 and method == 'drop'):
        for i in outlier_df.columns:
            x = 2
            h_outlier_df = outlier_df.drop(outlier_df[outlier_df[i] > x].index)

    elif (check == 3 and method == 'drop'):
        for i in outlier_df.columns:
            x = 3
            h_outlier_df = outlier_df.drop(outlier_df[outlier_df[i] > x].index)

    else:
        # it will drop the outlier values row\
        for i in outlier_df.columns:
            x = get_bound(outlier_df, i)[0]
            h_outlier_df = outlier_df.drop(outlier_df[outlier_df[i] > x].index)

    return h_outlier_df

def plot_graphs():
    pass 

