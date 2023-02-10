###########import packages##########
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
#%matplotlib 
###########import packages##########
import catboost
import xgboost
import lightgbm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import *
###########loading data##########
fdata=pd.read_csv('OER_activity_2ndsup.csv',encoding="gbk")
loo = LeaveOneOut()
not_needed_labels_list=['DOI','Country','Facility','Journal','Impact Factor','Published Date'
,'Cited Times','Number of days until 01/20/2023','Average cited times per day','Annealing Rising Rate degree min-1','O2 saturated/1 for yes'
,'Sample Name']
database_init=fdata.drop(labels=not_needed_labels_list,axis=1)
database_init.insert(0,'Metal_Dopant_1 Relative Atomic Mass',None)
database_init.insert(1,'Metal_Dopant_1 Atomic Number',None)
database_init.insert(2,'Metal_Dopant_1 Period',None)
database_init.insert(3,'Metal_Dopant_1 Group',None)
database_init.insert(4,'Metal_Dopant_1 Ionization Potential',None)
database_init.insert(5,'Metal_Dopant_1 Electronegativity',None)
database_init.insert(6,'Metal_Dopant_1 Number of d electrons',None)
database_init.insert(7,'Metal_Dopant_1 Atomic Radius',None)

database_init.insert(9,'Metal_Dopant_2 Relative Atomic Mass',None)
database_init.insert(10,'Metal_Dopant_2 Atomic Number',None)
database_init.insert(11,'Metal_Dopant_2 Period',None)
database_init.insert(12,'Metal_Dopant_2 Group',None)
database_init.insert(13,'Metal_Dopant_2 Ionization Potential',None)
database_init.insert(14,'Metal_Dopant_2 Electronegativity',None)
database_init.insert(15,'Metal_Dopant_2 Number of d electrons',None)
database_init.insert(16,'Metal_Dopant_2 Atomic Radius',None)

database_init.insert(18,'Metal_Dopant_3 Relative Atomic Mass',None)
database_init.insert(19,'Metal_Dopant_3 Atomic Number',None)
database_init.insert(20,'Metal_Dopant_3 Period',None)
database_init.insert(21,'Metal_Dopant_3 Group',None)
database_init.insert(22,'Metal_Dopant_3 Ionization Potential',None)
database_init.insert(23,'Metal_Dopant_3 Electronegativity',None)
database_init.insert(24,'Metal_Dopant_3 Number of d electrons',None)
database_init.insert(25,'Metal_Dopant_3 Atomic Radius',None)

database_init.insert(27,'Metal_Dopant_4 Relative Atomic Mass',None)
database_init.insert(28,'Metal_Dopant_4 Atomic Number',None)
database_init.insert(29,'Metal_Dopant_4 Period',None)
database_init.insert(30,'Metal_Dopant_4 Group',None)
database_init.insert(31,'Metal_Dopant_4 Ionization Potential',None)
database_init.insert(32,'Metal_Dopant_4 Electronegativity',None)
database_init.insert(33,'Metal_Dopant_4 Number of d electrons',None)
database_init.insert(34,'Metal_Dopant_4 Atomic Radius',None)

element_list=['Ag','Al','Au','B','Ba','Bi','Br','C','Ca','Cd','Ce','Cl','Co','Cr',
              'Cu','Er','Eu','F','Fe','Ga','Gd','Ho','In','Ir','K','La','Li','Lu',
              'Mg','Mn','Mo','N','Na','Nb','Ni','Os','P','Pb','Pd','Pr','Pt','Rh',
              'Ru','S','Sb','Sc','Se','Si','Sm','Sn','Sr','Ta','Tb','Te','Ti','Tm',
              'V','W','Y','Zn','Zr']
element_information={}

element_information['None']=[0,0,0,0,0,0,0,0]
element_information['none']=[0,0,0,0,0,0,0,0]
element_information['Sc']=[44.956, 21, 4, 3, 6.57, 1.36, 1, 162]
element_information['Ti']=[47.867, 22, 4, 4, 6.81, 1.54, 2, 147]
element_information['V']=[50.942, 23, 4, 5, 6.76, 1.63, 3, 134]
element_information['Cr']=[51.996, 24, 4, 6, 6.74, 1.66, 4, 130]
element_information['Mn']=[54.938, 25, 4, 7, 7.4, 1.56, 5, 127]
element_information['Fe']=[55.845, 26, 4, 8, 7.83, 1.83, 6, 126]
element_information['Co']=[58.933, 27, 4, 9, 7.81, 1.88, 7, 125]
element_information['Ni']=[58.6934, 28, 4, 10, 7.61, 1.91, 8, 124]
element_information['Cu']=[63.546, 29, 4, 11, 7.69, 1.9, 10, 128]
element_information['Zn']=[65.38, 30, 4, 12, 9.35, 1.65, 10, 138]
element_information['Ga']=[69.723, 31, 4, 13, 5.97, 1.81, 10, 141]
element_information['Y']=[88.905, 39, 5, 3, 6.5, 1.22, 1, 178]
element_information['Zr']=[91.224, 40, 5, 4, 6, 1.33, 2, 160]
element_information['Nb']=[92.906, 41, 5, 5, 6.76, 1.6, 4, 146]
element_information['Mo']=[95.94, 42, 5, 6, 7.35, 2.16, 5, 139]
element_information['Tc']=[98.906, 43, 5, 7, 7.28, 1.9, 5, 136]
element_information['Ru']=[101.07, 44, 5, 8, 7.5, 2.2, 7, 134]
element_information['Rh']=[102.905, 45, 5, 9, 7.7, 2.28, 8, 134]
element_information['Pd']=[106.42, 46, 5, 10, 8.3, 2.2, 10, 137]
element_information['Ag']=[107.868, 47, 5, 11, 7.54, 1.93, 10, 144]
element_information['Cd']=[112.411, 48, 5, 12, 8.95, 1.69, 10, 154]
element_information['In']=[114.818, 49, 5, 13, 8.95, 1.78, 10, 166]
element_information['Sn']=[118.71, 50, 5, 14, 7.37, 1.96, 10, 162]
element_information['La']=[138.905, 57, 6, 3, 5.5, 1.1, 1, 187]
element_information['Ce']=[140.116, 58, 6, 3, 6.91, 1.12, 1, 181]
element_information['Pr']=[140.904, 59, 6, 3, 5.76, 1.13, 10, 182]
element_information['Nd']=[144.242, 60, 6, 3, 6.31, 1.14, 10, 182]
element_information['Pm']=[144.912, 61, 6, 3 ,5.55, 1.13, 10, 183]
element_information['Sm']=[150.36, 62, 6, 3, 6.55, 1.1, 10, 181]
element_information['Eu']=[151.964, 63, 6, 3, 5.67, 1.2, 10, 199]
element_information['Gd']=[157.25, 64, 6, 3, 6.65, 1.2, 1, 179]
element_information['Tb']=[158.925, 65, 6, 3, 6.74, 1.2, 10, 180]
element_information['Dy']=[162.5, 66, 6, 3, 6.82, 1.22, 10, 180]
element_information['Ho']=[164.93, 67, 6, 3, 6.022, 1.23, 10, 179]
element_information['Er']=[167.529, 68, 6, 3, 6.108, 1.23, 10, 178]
element_information['Tm']=[168.934, 69, 6, 3, 6.184, 1.25, 10, 177]
element_information['Yb']=[173.04, 70, 6, 3, 7.06, 1.1, 10, 176]
element_information['Lu']=[174.967, 71, 6, 3, 5.4259, 1.27, 1, 175]
element_information['Hf']=[178.49, 72, 6, 4, 6.8251, 1.3, 2, 167]
element_information['Ta']=[180.947,73, 6, 5, 7.89, 1.5, 3, 149]
element_information['W']=[183.84, 74, 6, 6, 7.98, 2.36, 4, 141]
element_information['Re']=[186.207, 75, 6, 7, 7.88, 1.9, 5, 137]
element_information['Os']=[190.23, 76, 6, 8, 8.7, 2.2, 6, 135]
element_information['Ir']=[192.217, 77, 6, 9, 9.1, 2.2, 7, 136]
element_information['Pt']=[195.084, 78, 6, 10, 8.9, 2.28, 9, 139]
element_information['Au']=[196.966, 79, 6, 11, 9.19, 2.54, 10, 144]
element_information['Hg']=[200.59, 80, 6, 12, 10.39, 2, 10, 157]
element_information['Tl']=[204.383, 81, 6, 13, 6.08, 1.62, 10, 171]
element_information['Pb']=[207.2, 82, 6, 14, 7.38, 2.33, 10, 175]
element_information['Bi']=[208.98, 83, 6, 15, 7.25, 2.02, 10, 170]
###added
element_information['Ba']=[137.327, 56, 6, 2, 5.19, 0.89, 10, 222]
element_information['Sr']=[87.62, 38, 5, 2, 5.67, 0.95, 10, 215]
element_information['Na']=[22.9897, 11, 3, 1, 5.12, 0.93, 0, 190]
element_information['K']=[39.0983, 19, 4, 1, 4.32, 0.82, 0, 235]
element_information['Ca']=[40.078, 20, 4, 2, 6.09, 1, 0, 197]
element_information['Mg']=[24.3050, 12, 3, 2, 7.61, 1.31, 0, 160]
element_information['Li']=[6.941, 3, 2, 1, 5.37, 0.98, 0, 145]
element_information['C']=[12.0107, 6, 2, 14, 11.22, 2.56, 0, 77]
element_information['B']=[10.811, 5, 2, 13, 8.33, 2.04, 0, 98]
element_information['N']=[14.0067, 7, 2, 15, 14.48, 3.04, 0, 92]
element_information['P']=[30.9737, 15, 3, 15, 10.3, 2.19, 0, 128]
element_information['F']=[18.9984, 9, 2, 17, 18.6, 3.98, 0, 73]
element_information['S']=[32.065, 16, 3, 16, 10.31, 2.58, 0, 127]
element_information['Sb']=[121.760, 51, 5, 15, 8.35, 2.05, 10, 159]
element_information['Te']=[127.6, 52, 52, 5, 16, 9.0096, 2.1, 10, 160]
element_information['Br']=[79.904, 35, 4, 17, 11.8, 2.96, 10, 115]
element_information['Cl']=[35.453, 17, 3, 17, 12.96, 3.16, 0, 99]
element_information['Si']=[28.0855, 14, 3, 14, 8.12, 1.9, 0, 132]
element_information['Se']=[78.96, 34, 4, 16, 9.5, 2.55, 10, 140]
element_information['Al']=[26.9815, 13, 3, 13, 5.95, 1.61, 0,143]

for i in range(0,database_init.shape[0]):
    Metal_Dopant_1_type=database_init.iloc[i]['Metal_Dopant_1']
    Metal_Dopant_2_type=database_init.iloc[i]['Metal_Dopant_2']
    Metal_Dopant_3_type=database_init.iloc[i]['Metal_Dopant_3']
    Metal_Dopant_4_type=database_init.iloc[i]['Metal_Dopant_4']
    #####extract the element informations
    information_list_element_1=element_information[Metal_Dopant_1_type]
    information_list_element_2=element_information[Metal_Dopant_2_type]
    information_list_element_3=element_information[Metal_Dopant_3_type]
    information_list_element_4=element_information[Metal_Dopant_4_type]
#     print(information_list_element_1)
    #####input the element informations
    database_init.loc[i,'Metal_Dopant_1 Relative Atomic Mass']=information_list_element_1[0]
    database_init.loc[i,'Metal_Dopant_1 Atomic Number']=information_list_element_1[1]
    database_init.loc[i,'Metal_Dopant_1 Period']=information_list_element_1[2]
    database_init.loc[i,'Metal_Dopant_1 Group']=information_list_element_1[3]
    database_init.loc[i,'Metal_Dopant_1 Ionization Potential']=information_list_element_1[4]
    database_init.loc[i,'Metal_Dopant_1 Electronegativity']=information_list_element_1[5]
    database_init.loc[i,'Metal_Dopant_1 Number of d electrons']=information_list_element_1[6]
    database_init.loc[i,'Metal_Dopant_1 Atomic Radius']=information_list_element_1[7]
    
    database_init.loc[i,'Metal_Dopant_2 Relative Atomic Mass']=information_list_element_2[0]
    database_init.loc[i,'Metal_Dopant_2 Atomic Number']=information_list_element_2[1]
    database_init.loc[i,'Metal_Dopant_2 Period']=information_list_element_2[2]
    database_init.loc[i,'Metal_Dopant_2 Group']=information_list_element_2[3]
    database_init.loc[i,'Metal_Dopant_2 Ionization Potential']=information_list_element_2[4]
    database_init.loc[i,'Metal_Dopant_2 Electronegativity']=information_list_element_2[5]
    database_init.loc[i,'Metal_Dopant_2 Number of d electrons']=information_list_element_2[6]
    database_init.loc[i,'Metal_Dopant_2 Atomic Radius']=information_list_element_2[7]
    
    database_init.loc[i,'Metal_Dopant_3 Relative Atomic Mass']=information_list_element_3[0]
    database_init.loc[i,'Metal_Dopant_3 Atomic Number']=information_list_element_3[1]
    database_init.loc[i,'Metal_Dopant_3 Period']=information_list_element_3[2]
    database_init.loc[i,'Metal_Dopant_3 Group']=information_list_element_3[3]
    database_init.loc[i,'Metal_Dopant_3 Ionization Potential']=information_list_element_3[4]
    database_init.loc[i,'Metal_Dopant_3 Electronegativity']=information_list_element_3[5]
    database_init.loc[i,'Metal_Dopant_3 Number of d electrons']=information_list_element_3[6]
    database_init.loc[i,'Metal_Dopant_3 Atomic Radius']=information_list_element_3[7]
    
    database_init.loc[i,'Metal_Dopant_4 Relative Atomic Mass']=information_list_element_4[0]
    database_init.loc[i,'Metal_Dopant_4 Atomic Number']=information_list_element_4[1]
    database_init.loc[i,'Metal_Dopant_4 Period']=information_list_element_4[2]
    database_init.loc[i,'Metal_Dopant_4 Group']=information_list_element_4[3]
    database_init.loc[i,'Metal_Dopant_4 Ionization Potential']=information_list_element_4[4]
    database_init.loc[i,'Metal_Dopant_4 Electronegativity']=information_list_element_4[5]
    database_init.loc[i,'Metal_Dopant_4 Number of d electrons']=information_list_element_4[6]
    database_init.loc[i,'Metal_Dopant_4 Atomic Radius']=information_list_element_4[7]

database_2=database_init.drop(labels=['Metal_Dopant_1','Metal_Dopant_2','Metal_Dopant_3','Metal_Dopant_4'],axis=1)
database_final=database_2.fillna(database_2.median())
import pickle
database_final.to_pickle("./database_full_ac.pkl")