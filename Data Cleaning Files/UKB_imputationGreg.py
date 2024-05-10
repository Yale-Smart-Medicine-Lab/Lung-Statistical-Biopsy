#Imput missing data
# import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# # read in the saved imputer
# with (open('MaleImputer.pkl', 'rb')) as file:
#     maleImputer = pickle.load(file)

# # read in the data
# data = pd.read_csv('finalStatBioUKB.csv')

# # select just the male part of the data
# data_male = data[data['Gender']==1]
# columns = ['age', 'raceWh', 'raceBl', 'raceHis', 'raceAs', 'racePI', 'raceAI',
# 'raceMis', 'edlt8', 'ed8t11', 'ed12', 'edphs', 'edscol', 'edcol', 'edpcol',
# 'marmar', 'marwid', 'mardiv', 'marsep', 'marnvm', 'occhmk', 'occwrk', 'occune',
# 'occesl', 'occdis', 'occoth', 'bmi_curr', 'weight20', 'arthrit_f',
# 'bronchit_f', 'colon_comorbidity', 'diabetes_f', 'divertic_f', 'emphys_f',
# 'gallblad_f', 'hearta_f', 'hyperten_f', 'liver_comorbidity', 'osteopor_f',
# 'polyps_f', 'stroke_f', 'SMKEV', 'SMKNOW', 'cig_stop', 'pack_years', 'asppd',
# 'ibuppd', 'bili_fh', 'blad_fh', 'breast_fh', 'colo_fh', 'endo_fh', 'glio_fh',
# 'hnc_fh', 'hema_fh', 'liver_fh', 'lung_fh', 'mbreast_fh', 'mela_fh',
# 'ovarsumm_fh', 'panc_fh', 'pros_fh', 'renal_fh', 'thyd_fh', 'upgi_fh',
# 'trial_ph_bili', 'bili_is_first_dx', 'trial_ph_blad', 'blad_is_first_dx',
# 'trial_ph_colo', 'colo_is_first_dx', 'trial_ph_glio', 'glio_is_first_dx',
# 'trial_ph_hnc', 'hnc_is_first_dx', 'trial_ph_hema', 'hema_is_first_dx',
# 'trial_ph_liver', 'liver_is_first_dx', 'trial_ph_lung', 'lung_is_first_dx',
# 'trial_ph_mbreast', 'mbreast_is_first_dx', 'trial_ph_mela', 'mela_is_first_dx',
# 'trial_ph_panc', 'panc_is_first_dx', 'trial_ph_pros', 'pros_is_first_dx',
# 'trial_ph_renal', 'renal_is_first_dx', 'trial_ph_thyd', 'thyd_is_first_dx',
# 'trial_ph_upgi', 'upgi_is_first_dx', 'enlpros_f', 'enlprosa', 'infpros_f',
# 'infprosa', 'urinate_f', 'urinatea', 'surg_age', 'surg_biopsy',
# 'surg_prostatect', 'surg_resection', 'vasect_f', 'vasecta']
# data_male = data_male[columns]

# # imput missing data for males
# imputed_data_male = maleImputer.fit_transform(data_male)
# imputed_data_male = pd.DataFrame(imputed_data_male, columns=columns)
# pd.DataFrame(imputed_data_male).to_csv('imputedMaleData.csv', index=False)

# del maleImputer
# del data_male
# del imputed_data_male

# # read in the saved imputer
# with (open('femaleImputer.pkl', 'rb')) as file:
#     femaleImputer = pickle.load(file)
    
# # selet just the female part of the data
# data_female = data[data['Gender']==0]
# columns = ['age', 'raceWh', 'raceBl', 'raceHis', 'raceAs', 'racePI', 'raceAI',
# 'raceMis', 'edlt8', 'ed8t11', 'ed12', 'edphs', 'edscol', 'edcol', 'edpcol',
# 'marmar', 'marwid', 'mardiv', 'marsep', 'marnvm', 'occhmk', 'occwrk', 'occune',
# 'occesl', 'occdis', 'occoth', 'bmi_curr', 'weight20', 'arthrit_f',
# 'bronchit_f', 'colon_comorbidity', 'diabetes_f', 'divertic_f', 'emphys_f',
# 'gallblad_f', 'hearta_f', 'hyperten_f', 'liver_comorbidity', 'osteopor_f',
# 'polyps_f', 'stroke_f', 'SMKEV', 'SMKNOW', 'cig_stop', 'pack_years', 'asppd',
# 'ibuppd', 'bili_fh', 'blad_fh', 'breast_fh', 'colo_fh', 'endo_fh', 'glio_fh',
# 'hnc_fh', 'hema_fh', 'liver_fh', 'lung_fh', 'mbreast_fh', 'mela_fh',
# 'ovarsumm_fh', 'panc_fh', 'pros_fh', 'renal_fh', 'thyd_fh', 'upgi_fh',
# 'trial_ph_bili', 'bili_is_first_dx', 'trial_ph_blad', 'blad_is_first_dx',
# 'trial_ph_colo', 'colo_is_first_dx', 'trial_ph_glio', 'glio_is_first_dx',
# 'trial_ph_hnc', 'hnc_is_first_dx', 'trial_ph_hema', 'hema_is_first_dx',
# 'trial_ph_liver', 'liver_is_first_dx', 'trial_ph_lung', 'lung_is_first_dx',
# 'trial_ph_mbreast', 'mbreast_is_first_dx', 'trial_ph_mela', 'mela_is_first_dx',
# 'trial_ph_panc', 'panc_is_first_dx', 'trial_ph_pros', 'pros_is_first_dx',
# 'trial_ph_renal', 'renal_is_first_dx', 'trial_ph_thyd', 'thyd_is_first_dx',
# 'trial_ph_upgi', 'upgi_is_first_dx', 'hystera', 'ovariesr_f', 'tubllig',
# 'bcontra_f', 'bcontra', 'curhorm', 'horm_f', 'thorm', 'fchilda', 'livec',
# 'miscar', 'preg_f', 'stillb', 'trypreg', 'tubal', 'fmenstr', 'bbd',
# 'benign_ovcyst', 'endometriosis', 'uterine_fib']
# data_female = data_female[columns]

# # imput missing data for females
# imputed_data_female = femaleImputer.fit_transform(data_female)
# imputed_data_female = pd.DataFrame(imputed_data_female, columns=columns)
# pd.DataFrame(imputed_data_female).to_csv('imputedFemaleData.csv', index=False)

# read in the data
print('Reading in UK Biobank data')
data = pd.read_csv('Data Files/finalStatBioUKB.csv')
print('Reading in male PLCO data')
maleData = pd.read_csv('Male_Lung_Data_Greg_imputed.csv')
means = maleData.mean()
UKBiobankData = data[data['gender']==0]
UKBiobankData = UKBiobankData[maleData.columns]

print('Taking care of empty columns')
# UKBiobankData = pd.concat([UKBiobankData, means])
columnWNoVar = UKBiobankData.columns[np.isnan(UKBiobankData.std())].to_list()
for column in columnWNoVar:
    if np.isnan(means[column]):
        UKBiobankData[column] = 0
    else:
        UKBiobankData[column] = means[column]
columnWNoVar  = UKBiobankData.columns[UKBiobankData.std()==0].to_list()
# UKBiobankData = pd.append([UKBiobankData, means])
for column in columnWNoVar:
    UKBiobankData.loc[UKBiobankData[column].isna(), column] = UKBiobankData[column].mean()
print('Beginning imputation')
imputer = KNNImputer(n_neighbors=5)
imputed_UKBiobankData = imputer.fit_transform(UKBiobankData)
imputed_UKBiobankData = pd.DataFrame(imputed_UKBiobankData, columns=UKBiobankData.columns)#[~UKBiobankData.columns.isin(columnWNoVar)])
# imputed_UKBiobankData = imputed_UKBiobankData.iloc[:-1]
# for column in columnWNoVar:
#     imputed_UKBiobankData[column] = 0
print('Saving data to file')
pd.DataFrame(imputed_UKBiobankData).to_csv('imputedMaleDataGreg.csv', index=False)
del maleData

print('Reading in male PLCO data')
femaleData = pd.read_csv('Female_Lung_Data_Greg_imputed.csv')
means = femaleData.mean()
UKBiobankData = data[data['gender']==1]
UKBiobankData = UKBiobankData[femaleData.columns]

print('Taking care of empty columns')
#columnWNoVar  = UKBiobankData.columns[UKBiobankData.std()==0].to_list()
columnWNoVar = UKBiobankData.columns[np.isnan(UKBiobankData.std())].to_list()
for column in columnWNoVar:
    if np.isnan(means[column]):
        UKBiobankData[column] = 0
    else:
        UKBiobankData[column] = means[column]
columnWNoVar  = UKBiobankData.columns[UKBiobankData.std()==0].to_list()
# UKBiobankData = pd.append([UKBiobankData, means])
for column in columnWNoVar:
    UKBiobankData.loc[UKBiobankData[column].isna(), column] = UKBiobankData[column].mean()
print('Beginning imputation')
imputer = KNNImputer(n_neighbors=5)
imputed_UKBiobankData = imputer.fit_transform(UKBiobankData)
imputed_UKBiobankData = pd.DataFrame(imputed_UKBiobankData, columns=UKBiobankData.columns)#[~UKBiobankData.columns.isin(columnWNoVar)])
# imputed_UKBiobankData = imputed_UKBiobankData.iloc[:-1]
# for column in columnWNoVar:
#     if column == 'gender':
#         imputed_UKBiobankData[column] = 1
#     else:
#         imputed_UKBiobankData[column] = 0
print('Saving data to file')
pd.DataFrame(imputed_UKBiobankData).to_csv('imputedFemaleDataGreg.csv', index=False)