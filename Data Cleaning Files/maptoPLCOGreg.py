# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:30:12 2021

@author: ghart
"""

import pandas as pd
import numpy as np
import datetime

raw_data = pd.read_csv('Data Files/rawStatBioUKB.csv')
reduced_data = pd.DataFrame()

# adding sex splitting the data in the future
# UKBiobank 0-Female, 1-Male
# PLCO      0-Male,   1-Female
reduced_data['gender'] = 1 - raw_data['31-0.0']

# age at recruitment
age = raw_data['21022-0.0']
reduced_data['age'] = age/100

# Ethnic background
# guessing at how to best match this to PLCO - I going to put mixed races as
# non-white, because I think that is the way the US does it
# PLCO -- Biobank
# 1 (white) -- 1 (white), 1001 (British), 1002 (Irish), 1003 (Other white)
# 2 (black) -- 2001 (White and Black Caribbean), 2002 (White and Black African),
#              4002 (African), 4003 (Other Black), 4 (Black), 4001 (Caribbean)
# 3 (Hispanic) -- None?
# 4 (Asian) -- 3001 (Indian), 3002 (Pakistani), 2003 (White and Asian),
#              3 (Asian or Asian British), 3003 (Bangladeshi), 3004 (Other Asian),
#              5 (Chinese)
# 5 (Pacific Islander) -- None?
# 6 (American Indian) -- None?
# ? -- 2 (Mixed), 2004 (Other Mixed), 6 (Other Ethnic Group)
plco_field = 'race7'
reduced_data[plco_field] = 7
plco_codes = [1, 2, 4]
ukbb_codes = [[1, 1001, 1002, 1003], [2001, 2002, 4002, 4003, 4, 4001],
              [3001, 3002, 2003, 3, 3003, 3004, 5]]
instances = ['0.0', '1.0','2.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['21000-'+instance]==old_code, reduced_data[plco_field]==7)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data['raceWh']  = reduced_data[plco_field] == 1
reduced_data['raceBl']  = reduced_data[plco_field] == 2
reduced_data['raceHis'] = reduced_data[plco_field] == 3
reduced_data['raceAs']  = reduced_data[plco_field] == 4
reduced_data['racePI']  = reduced_data[plco_field] == 5
reduced_data['raceAI']  = reduced_data[plco_field] == 6
reduced_data['raceMis'] = reduced_data[plco_field] == 7
# Do we check if the other instances can fill in any missingness?
# What to do about the used groups from UKBB?
        
# Education level
# trying to match to US system, again I am mosly guessing
# PLCO -- Biobank
# 1 (< 8 years) -- None?
# 2 (2-11 Years) -- None?
# 3 (Completted High School) -- 3 (O levels), 4 (CSE)
# 4 (Post High School non-college) -- 6 (other professional)
# 5 (Some College) -- 2 (A levels), 5 (NVQ)
# 6 (College Graduate) -- 1 (College or University degree)
# 7 (Postgraduate) -- None?
plco_field = 'education'
reduced_data[plco_field] = 9
plco_codes = [3, 4, 5, 6]
ukbb_codes = [[3, 4], [6],
              [2, 5], [1]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['6138-'+instance]==old_code, reduced_data[plco_field]==9)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data['edlt8']  = reduced_data[plco_field] == 1
reduced_data['ed8t11'] = reduced_data[plco_field] == 2
reduced_data['ed12']   = reduced_data[plco_field] == 3
reduced_data['edphs']  = reduced_data[plco_field] == 4
reduced_data['edscol'] = reduced_data[plco_field] == 5
reduced_data['edcol']  = reduced_data[plco_field] == 6
reduced_data['edpcol'] = reduced_data[plco_field] == 7
#reduced_data['edmis']  = reduced_data[plco_field] == 9
        
# Marriage status
# I can only find if they are married/living together if not we do not know if 
# they are divorced, widowed, never married, etc.
plco_field = 'marriage'
reduced_data[plco_field] = 9
plco_codes = [1]
ukbb_codes = [[1]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['6141-'+instance]==old_code, reduced_data[plco_field]==9)
            reduced_data.loc[idx, plco_field] = new_code        
reduced_data['marmar'] = reduced_data[plco_field] == 1
reduced_data['marwid'] = reduced_data[plco_field] == 2
reduced_data['mardiv'] = reduced_data[plco_field] == 3
reduced_data['marsep'] = reduced_data[plco_field] == 4
reduced_data['marnvm'] = reduced_data[plco_field] == 5
#reduced_data['marmis'] = reduced_data[plco_field] == 9
            
# Occupation
# PLCO -- Biobank
# 1 (Homemaker) -- 3 (Looking after home and/or family)
# 2 (Working) -- 1 (In paid employment or self-employed)
# 3 (Unemployed) -- 5 (Unemployed)
# 4 (Retired) -- 2 (Retired)
# 5 (Extended Sick leave) -- None?
# 6 (Disabled) -- 4 (Unable to work because of sickness or disability)
# 7 (Other) -- 6 (Doing unpaid or voluntary work), 7 (Full or part-time student)
plco_field = 'occupation'
reduced_data[plco_field] = 9
plco_codes = [1, 2, 3, 4, 6, 7]
ukbb_codes = [[3], [1], [5],
              [2], [4], [6, 7]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5',
             '0.6', '1.6', '2.6', '3.6']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['6142-'+instance]==old_code, reduced_data[plco_field]==9)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data['occhmk'] = reduced_data[plco_field] == 1
reduced_data['occwrk'] = reduced_data[plco_field] == 2
reduced_data['occune'] = reduced_data[plco_field] == 3
reduced_data['occret'] = reduced_data[plco_field] == 4
reduced_data['occesl'] = reduced_data[plco_field] == 5
reduced_data['occdis'] = reduced_data[plco_field] == 6
reduced_data['occoth'] = reduced_data[plco_field] == 7
#reduced_data['occmis'] = reduced_data[plco_field] == 9

print('Done with demographics')            
# BMI
# PLCO
# 1 (<18.5)
# 2 (18.5-25)
# 3 (25-30)
# 4 (>30)
plco_field = 'bmi_curr'
reduced_data[plco_field] = np.nan
plco_codes = [1, 2, 3, 4]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = np.isnan( reduced_data[plco_field])
    reduced_data.loc[idx, plco_field] = raw_data['21001-'+instance]
reduced_data[plco_field] = reduced_data[plco_field]/100
    
# Weight at 20
# not in Biobank
plco_field = 'weight20_f'
reduced_data[plco_field] = np.nan

# Arthritis
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'arthrit_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['M0000','M0001','M0002','M0003','M0005','M0006','M0007','M0008','M0009',
                   'M0010','M0012','M0016',
                   'M0020','M0021','M0023','M0024','M0025','M0026','M0027','M0028','M0029',
                   'M0080','M0081','M0085','M0086','M0087',
                   'M0090','M0091','M0092','M0093','M0094','M0095','M0096','M0097','M0098','M0099',
                   'M0107',
                   'M011',
                   'M0139',
                   'M0150',
                   'M0186','M0187','M0189',
                   'M0500','M0509',
                   'M0510','M0517','M0518','M0519',
                   'M0520','M0524','M0526','M0528','M0529',
                   'M0530','M0538','M0539',
                   'M0580','M0582','M0583','M0584','M0586','M0587','M0588','M0589',
                   'M0590','M0591','M0592','M0593','M0594','M0595','M0596','M0597','M0598','M0599',
                   'M0600','M0601','M0602','M0603','M0604','M0605','M0606','M0607','M0608','M0609',
                   'M0610','M0611','M0616','M0619',
                   'M0621','M0622','M0625','M0626','M0627',
                   'M0630','M0632','M0633','M0634','M0636','M0637','M0638','M0639',
                   'M0640','M0641','M0643','M0644','M0645','M0646','M0647','M0649',
                   'M0680','M0681','M0682','M0684','M0685','M0686','M0687','M0688','M0689',
                   'M0690','M0691','M0692','M0693','M0694','M0695','M0696','M0697','M0698','M0699',
                   'M0717','M0719',
                   'M0800','M0802','M0804','M0805','M0806','M0807','M0808','M0809',
                   'M0810','M0819',
                   'M0820','M0822','M0825','M0826','M0827','M0828','M0829',
                   'M0830','M0836','M0839',
                   'M0840','M0844',
                   'M0880',
                   'M0890','M0896','M0899',
                   'M0909',
                   'M1300','M1301','M1303','M1304','M1305','M1306','M1307','M1308','M1309',
                   'M1310','M1311','M1312','M1313','M1314','M1315','M1316','M1317','M1319',
                   'M1380','M1381','M1382','M1383','M1384','M1385','M1386','M1387','M1388','M1389',
                   'M1390','M1391','M1392','M1393','M1394','M1395','M1396','M1397','M1398','M1399']]
cond_len = 213
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            
# Bronchitis
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'bronchit_f'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['22129-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code

# Colon comorbidity
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'colon_comorb'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['K510','K511','K512','K513','K514','K515','K518','K519',
                   'K500','K501','K508','K509']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Diabetes
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'diabetes_f'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['2443-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code

# Divertic
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'divertic_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['K572','K573']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Emphysema
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'emphys_f'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['22128-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code
            
# Gallblader
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'gallblad_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['K800','K801','K802','K803','K8036',
                   'K804','K805','K808','K810','K811',
                   'K818','K819']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Heart disease
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'hearta_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['I210','I211','I212','I213','I214','I219',
                   'I220','I221','I228','I229',
                   'I240','I241','I248','I249',
                   'I250','I251','I252','I253','I254','I255','I256','I258','I259']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Hypertension
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'hyperten_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['I10']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Liver comorbidity
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'liver_comorb'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['B159','B160','B169',
                   'B170','B171','B172','B178','B179',
                   'B180','B181','B182','B188','B189',
                   'B199','K703','K717',
                   'K740','K741','K742','K743','K744','K745','K746']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Osteoporosis
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'osteopor_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['M8000','M8001','M8003','M8005','M8007','M8008','M8009',
                   'M8010','M8021','M8025','M8028',
                   'M8040','M8041','M8046','M8048','M8049',
                   'M8050','M8052','M8053','M8055','M8058',
                   'M8080','M8081','M8082','M8085','M8086','M8087','M8088','M8089',
                   'M8090','M8091','M8092','M8093','M8094','M8095','M8096','M8097','M8098','M8099',
                   'M8100','M8102','M8105','M8108','M8109',
                   'M8110','M8119','M8120','M8129',
                   'M8130','M8138','M8139',
                   'M8140','M8143','M8144','M8145','M8147','M8148','M8149',
                   'M8150','M8151','M8155','M8158','M8159',
                   'M8165','M8167','M8168','M8169',
                   'M8180','M8185','M8187','M8188','M8189',
                   'M8190','M8191','M8192','M8193','M8194','M8195','M8196','M8197','M8198','M8199',
                   'M8200','M8201','M8206','M8208','M8209',
                   'M8210','M8211','M8213','M8219']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Colorectal polyps
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'polyps_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['K620','K621','K635']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Stroke
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'stroke_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['G463','G464','I64','I694']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

print('Starting smoking questions')
# Ever smoked
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'SMKEV'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20160-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code

# Currently smoked
# PLCO -- Biobank
# 0 (No) -- 0 (Never), 1 (Previous)
# 1 (Yes) -- 2 (Current)
plco_field = 'SMKNOW'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0,1], [2]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20116-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code

# Number of years since Stopped Smoking
plco_field = 'cig_stop'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[0], [2]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = reduced_data[plco_field]==0
    reduced_data.loc[idx, plco_field] = (age - raw_data['6194-'+instance])[idx]
reduced_data[plco_field] = reduced_data[plco_field]/100
    
# Number of pack years somked
plco_field = 'pack_years'
reduced_data[plco_field] = np.nan
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = np.isnan(reduced_data[plco_field])
    reduced_data.loc[idx, plco_field] = raw_data['20161-'+instance][idx]
reduced_data[plco_field] = reduced_data[plco_field]/500


# Taking aspirin
# Biobank only has if it is taken regularly.
# PLCO -- Biobank
plco_field = 'asppd'
reduced_data[plco_field] = 0
plco_codes = [1,9]
ukbb_codes = [[1], [-1, -3]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['6154-'+instance]==old_code, reduced_data[plco_field]==9)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data[plco_field] = reduced_data[plco_field]/60

# Taking ibuprofen
# Biobank only has if it is taken regularly.
# PLCO -- Biobank
plco_field = 'ibuppd'
reduced_data[plco_field] = 0
plco_codes = [1,9]
ukbb_codes = [[2], [-1, -3]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['6154-'+instance]==old_code, reduced_data[plco_field]==9)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data[plco_field] = reduced_data[plco_field]/60

# Family history of cancer (prostate)
# Done per family member
# PLCO -- Biobank
plco_field = 'pros_fh'
reduced_data[plco_field] = 0
reduced_data[plco_field + '_age'] = np.nan
reduced_data[plco_field + '_cnt'] = 0
plco_codes = [1,9]
ukbb_codes = [[13], [-11, -13, -21, -23]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5',
             '0.6', '1.6', '2.6', '3.6',
             '0.7', '1.7', '2.7', '3.7',
             '0.8', '1.8', '2.8', '3.8',
             '0.9', '1.9', '2.9', '3.9']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20107-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.10', '1.10', '2.10', '3.10']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.11', '1.11', '2.11', '3.11']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
reduced_data.loc[reduced_data[plco_field]==9,plco_field + '_cnt'] = np.nan
reduced_data.loc[reduced_data[plco_field]==9,plco_field] = np.nan
reduced_data[plco_field + '_cnt'] = reduced_data[plco_field + '_cnt']/10

# Family history of cancer (breast)
# Done per family member
# PLCO -- Biobank
plco_field = 'breast_fh'
reduced_data[plco_field] = 0
reduced_data[plco_field + '_age'] = np.nan
reduced_data[plco_field + '_cnt'] = 0
plco_codes = [1,9]
ukbb_codes = [[5], [-11, -13, -21, -23]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5',
             '0.6', '1.6', '2.6', '3.6',
             '0.7', '1.7', '2.7', '3.7',
             '0.8', '1.8', '2.8', '3.8',
             '0.9', '1.9', '2.9', '3.9']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20107-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.10', '1.10', '2.10', '3.10']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.11', '1.11', '2.11', '3.11']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
reduced_data.loc[reduced_data[plco_field]==9,plco_field + '_cnt'] = np.nan
reduced_data.loc[reduced_data[plco_field]==9,plco_field] = np.nan
reduced_data[plco_field + '_cnt'] = reduced_data[plco_field + '_cnt']/10

# Family history of cancer (bowel)
# Done per family member
# PLCO -- Biobank
plco_field = 'colo_fh'
reduced_data[plco_field] = 0
reduced_data[plco_field + '_age'] = np.nan
reduced_data[plco_field + '_cnt'] = 0
plco_codes = [1,9]
ukbb_codes = [[4], [-11, -13, -21, -23]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5',
             '0.6', '1.6', '2.6', '3.6',
             '0.7', '1.7', '2.7', '3.7',
             '0.8', '1.8', '2.8', '3.8',
             '0.9', '1.9', '2.9', '3.9']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20107-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.10', '1.10', '2.10', '3.10']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.11', '1.11', '2.11', '3.11']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
reduced_data.loc[reduced_data[plco_field]==9,plco_field + '_cnt'] = np.nan
reduced_data.loc[reduced_data[plco_field]==9,plco_field] = np.nan
reduced_data[plco_field + '_cnt'] = reduced_data[plco_field + '_cnt']/10

# Family history of cancer (Lung)
# Done per family member
# PLCO -- Biobank
plco_field = 'lung_fh'
reduced_data[plco_field] = 0
reduced_data[plco_field + '_age'] = np.nan
reduced_data[plco_field + '_cnt'] = 0
plco_codes = [1,9]
ukbb_codes = [[3], [-11, -13, -21, -23]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3',
             '0.4', '1.4', '2.4', '3.4',
             '0.5', '1.5', '2.5', '3.5',
             '0.6', '1.6', '2.6', '3.6',
             '0.7', '1.7', '2.7', '3.7',
             '0.8', '1.8', '2.8', '3.8',
             '0.9', '1.9', '2.9', '3.9']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20107-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.10', '1.10', '2.10', '3.10']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20110-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
instances = ['0.11', '1.11', '2.11', '3.11']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['20111-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            if new_code==1:
                reduced_data.loc[idx, plco_field + '_cnt'] += 1
reduced_data.loc[reduced_data[plco_field]==9,plco_field + '_cnt'] = np.nan
reduced_data.loc[reduced_data[plco_field]==9,plco_field] = np.nan
reduced_data[plco_field + '_cnt'] = reduced_data[plco_field + '_cnt']/10

cancers = ['bili','blad','endo','glio','hnc','hema','liver','mbreast','mela',
           'ovarsumm','panc','renal','thyd','upgi']
for cancer in cancers:
    reduced_data[cancer + '_fh'] = np.nan
    reduced_data[cancer + '_fh_age'] = np.nan
    reduced_data[cancer + '_fh_cnt'] = np.nan

cancers = ['lung']
for cancer in cancers:
        reduced_data['trial_ph_' + cancer] = np.nan
        reduced_data[cancer + '_is_first_dx'] = np.nan

print('Working on cancers')
# Personal history of cancer
# Check for cancer and cancer age
cancers = ['lung']
ukbb_codes = {'bili': ['C23', 'C230',
                          'C240','C241','C248','C249'],
              'blad': ['C670','C671','C672','C673','C674','C675','C676','C677','C678','C679',
                          'D090'],
              'breast': ['C500','C501','C502','C503','C504','C505','C506','C508','C509',
                         'D050','D051','D057','D059'],
              'colo': ['C180','C181','C182','C183','C184','C185','C186','C187','C188','C189',
                             'C19','C20','C190','C200',
                             'C210','C211','C218',
                             'C785',
                             'D010','D011','D012','D013','D014'],
              'endo': ['C540','C541','C542','C543','C549',
                              'C55', 'C550',
                              'D070'],
              'glio': ['C710','C711','C712','C713','C714','C715','C716','C717','C718','C719', # These are all brian cancers can't tell what is glioma
                         'C793'], 
              'hnc': ['C000','C001','C002','C003','C004','C005','C006','C009',
                           'C01','C010',
                           'C020','C021','C022','C023','C024','C028','C029',
                           'C030','C031','C039',
                           'C040','C041','C048','C049',
                           'C050','C051','C052','C058','C059',
                           'C060','C061','C062','C068','C069',
                           'C07','C070',
                           'C080','C081','C089',
                           'C090','C091','C098','C099',
                           'C100','C101','C102','C103','C104','C108','C109',
                           'C110','C111','C118','C119',
                           'C12','C120',
                           'C130','C131','C132','C139',
                           'C140','C148',
                           'C150','C151','C152','C153','C154','C155','C158','C159', 'D001',# adding oespagus to Neck not Upper GI
                           'C300','C301',
                           'C310','C311','C319',
                           'C320','C321','C322','C323','C328','C329',
                           'C760',
                           'D000'],
              'hema': ['C420','C421','C422','C423','C424',
                                'C810','C811','C812','C813','C817','C819',
                                'C820','C821','C822','C827','C829',
                                'C830','C831','C832','C833','C834','C835','C837','C838','C839',
                                'C840','C841','C842','C843','C844','C845',
                                'C850','C850','C850','C850',
                                'C862',
                                'C880','C884','C889',
                                'C900','C901','C902','C903',
                                'C910','C911','C913','C914','C915','C919',
                                'C920','C921','C923','C924','C925','C927','C929',
                                'C930','C931',
                                'C940','C944','C946',
                                'C950','C951','C957','C959',
                                'C961','C962','C963','C967','C968','C969'],
              'liver': ['C220','C221','C223','C224','C227','C229',
                        'C787','D015'], #D015 is both liver and biliary
              'lung': ['C33','C330'
                       'C340','C340','C340','C340','C340','C340',
                       'C381','C382','C383','C384', 'C398', 'C399', # do not know if these should be included
                       'C780','C781','C782','C783', # do not know if we should inclued all these 
                       'D020','D022','D023'], 
              'mela': ['C430','C431','C432','C433','C434','C435','C436','C437','C438','C439',
                           'D030','D031','D032','D033','D034','D035','D036','D037','D038','D039'],
              'osumm': ['C56','C560',
                          'C796'],
              'panc': ['C250','C251','C252','C253','C254','C257','C258','C259'],
              'pros': ['C61','C610',
                           'D075'],
              'renal': ['C64','C640',
                        'C65','C650',
                        'C66','C660',
                        'C790'],
              'thyd': ['C73','C730',
                          'D093'], # includes more than Thyroid
              'upgi': ['C160','C161','C162','C163','C164','C165','C166','C168','C169',
                          'C170','C171','C172','C178','C179',
                          'C784',
                          'D002']}

# We only have access to birth year and month
birth_year = '34-0.0'
birth_month = '52-0.0'
birth_year = np.array(raw_data[birth_year]).astype(int)
birth_month = np.array(raw_data[birth_month]).astype(int)
birth_year[birth_year<1] = 1900
birth_month[birth_month<1] = 1
birth_date = [datetime.datetime.strptime(str(year)+'-'+str(month)+'-15', '%Y-%m-%d').date() for year, month in zip(birth_year, birth_month)]
NA_date = datetime.datetime(1900,1,1).date()
dia_after_enroll = np.ones((len(cancers),len(age)))*100
for i, cancer in enumerate(cancers):
    plco_field = 'trial_ph_' + cancer
    plco_field_age = cancer + "_age"
    plco_field_first = cancer + '_is_first_dx'
    
    reduced_data[plco_field] = 0
    reduced_data[plco_field_age] = 0
    reduced_data[plco_field_first] = 0
    
    dia_date = np.array([NA_date] * len(reduced_data[plco_field]))  # Ensure dia_date starts as datetime objects
    
    for instance in range(cond_len):
        for old_code in ukbb_codes[cancer]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)] == old_code, dia_date == NA_date)
            dates_to_convert = raw_data.loc[idx, '41280-0.' + str(instance)]
            converted_dates = pd.to_datetime(dates_to_convert, format='%Y-%m-%d', errors='coerce').dt.date
            dia_date[idx] = converted_dates
            reduced_data.loc[idx, plco_field] = 1

    if pd.api.types.is_datetime64_any_dtype(dia_date):
        reduced_data[plco_field_age] = dia_date
        reduced_data[plco_field_age] = (reduced_data[plco_field_age] - birth_date).dt.days / 365.24
    else:
        print("Error: dia_date is not datetime type")

    dia_after_enroll[i, :] = reduced_data[plco_field_age] - age
    dia_after_enroll[i, age > reduced_data[plco_field_age]] = 100



    
idx = np.argmin(dia_after_enroll, axis=0)
for i, cancer in enumerate(cancers):
    plco_field_first = cancer + '_is_first_dx'
    reduced_data[plco_field_first] = idx == i
    
cancers = ['lung']
for i, cancer in enumerate(cancers):
    reduced_data[cancer] = dia_after_enroll[i,:]<5.0
#reduced_data['mbreast'] = reduced_data['breast']
  
print('Female specific questions')
# Age at hysterectomy
# PLCO -- Biobank
# 1 (<40) --
# 2 (40-45) --
# 3 (45-50) --
# 4 (50-55) --
# 5 (>=55) -- 
plco_field = 'hystera'
reduced_data[plco_field] = np.nan
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    field = '2824-'+instance
    idx = np.logical_and(raw_data[field]<40, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(np.logical_and(raw_data[field]>=40, raw_data[field]<45),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 2
    idx = np.logical_and(np.logical_and(raw_data[field]>=45, raw_data[field]<50),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 3
    idx = np.logical_and(np.logical_and(raw_data[field]>=50, raw_data[field]<55),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 4
    idx = np.logical_and(raw_data[field]>=55, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 5
reduced_data[plco_field] = reduced_data[plco_field] / 5
    
# Removed ovaries
# PLCO -- Biobank
# 0 (Ovaries Not Removed) -- If not others
# 1 (One Ovary - Partial) -- 
# 2 (One Ovary - Total) --
# 3 (Both Ovaries - Partial) --
# 4 (Both Ovaries - Total) --
# 5 (Don't Know) --
# 8 (Ambiguous) --
plco_field = 'ovariesr_f'
reduced_data[plco_field] = 0
plco_codes = [4, 2, 8, 1]
ukbb_codes = [['Q221', 'Q223', 'Q232','Q236'], 
              ['Q231', 'Q235'], ['Q241','Q248'],
              ['Q431','Q432','Q433','Q438','Q439']]
surg_len = 117
for instance in range(surg_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data.loc[reduced_data[plco_field]==8, plco_field] = np.nan
reduced_data[plco_field] = reduced_data[plco_field]/4


# tubal ligation
# PLCO -- Biobank
# 0 (No) -- If not others
# 1 (Yes) -- 
# 2 (Don't Know) --
plco_field = 'tuballig'
reduced_data[plco_field] = 0
plco_codes = [1]
ukbb_codes = [['Q282', 'Q281', 'Q271']]
for instance in range(surg_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Every used birth control
# PLCO -- Biobank
# 0 (No) -- If not others
# 1 (Yes) -- 
plco_field = 'bcontr_f'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['2784-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code

# Age started birth control
# PLCO -- Biobank
# 1 (<30) --
# 2 (30-39) --
# 3 (40-49) --
# 4 (>50) -- 
plco_field = 'bcontra'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = np.logical_and(raw_data['2794-'+instance]<30, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(np.logical_and(raw_data['2794-'+instance]>=30, raw_data['2794-'+instance]<40),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 2
    idx = np.logical_and(np.logical_and(raw_data['2794-'+instance]>=40, raw_data['2794-'+instance]<=50),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 3
    idx = np.logical_and(raw_data['2794-'+instance]>50, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 4
reduced_data[plco_field] = reduced_data[plco_field]/4
    
# Currently taking hormones
# PLCO -- Biobank
# 0 (No) -- If not others
# 1 (Yes) -- 
plco_field = 'curhorm'
reduced_data[plco_field] = 0
plco_codes = [1, 9]
ukbb_codes = [[4], [-1, -3]]
instances = ['0.0', '1.0', '2.0', '3.0',
             '0.1', '1.1', '2.1', '3.1',
             '0.2', '1.2', '2.2', '3.2',
             '0.3', '1.3', '2.3', '3.3']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['6153-'+instance]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Ever used hormone-replacement therapy
# PLCO -- Biobank
# 0 (No) -- 
# 1 (Yes) -- 
# 2 (Don't Know) --
plco_field = 'horm_f'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1, 2]
ukbb_codes = [[0], [1], [-1]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['2814-'+instance]==old_code, np.isnan(reduced_data[plco_field]))
            reduced_data.loc[idx, plco_field] = new_code
            
# How many years on HRT
# PLCO -- Biobank
# 0 (Not Applicable) -- 
# 1 (>=10) -- 
# 2 (10-6) --
# 3 (6-4) --
# 4 (4-1) --
# 5 (<=1) --
plco_field = 'thorm'
reduced_data[plco_field] = 0
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    field_start = '3536-'+instance
    field_end = '3546-'+instance
    idx = np.logical_and((raw_data[field_end]-raw_data[field_start])>=10, reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(np.logical_and((raw_data[field_end]-raw_data[field_start])<10, (raw_data[field_end]-raw_data[field_start])>=6),
                         reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 2
    idx = np.logical_and(np.logical_and((raw_data[field_end]-raw_data[field_start])<6, (raw_data[field_end]-raw_data[field_start])>=4),
                         reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 3
    idx = np.logical_and(np.logical_and((raw_data[field_end]-raw_data[field_start])<4, (raw_data[field_end]-raw_data[field_start])>=2),
                         reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 4
    idx = np.logical_and((raw_data[field_end]-raw_data[field_start])<2, reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 5
reduced_data[plco_field] = 6 - reduced_data[plco_field]
reduced_data.loc[reduced_data[plco_field]==6, plco_field] = 0
reduced_data[plco_field] = reduced_data[plco_field]/5

# Age at first child
# PLCO -- Biobank 
# 1 (<16) -- 
# 2 (16-20) --
# 3 (20-25) --
# 4 (25-30) --
# 5 (30-35) --
# 6 (35-40) --
# 7 (>=40) --
plco_field = 'fchilda'
reduced_data[plco_field] = np.nan
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = np.logical_and(raw_data['3872-'+instance]<16, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(np.logical_and(raw_data['3872-'+instance]>=16, raw_data['3872-'+instance]<20),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 2
    idx = np.logical_and(np.logical_and(raw_data['3872-'+instance]>=20, raw_data['3872-'+instance]<25),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 3
    idx = np.logical_and(np.logical_and(raw_data['3872-'+instance]>=25, raw_data['3872-'+instance]<30),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 4
    idx = np.logical_and(np.logical_and(raw_data['3872-'+instance]>=30, raw_data['3872-'+instance]<35),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 5
    idx = np.logical_and(np.logical_and(raw_data['3872-'+instance]>=35, raw_data['3872-'+instance]<40),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 6
    idx = np.logical_and(raw_data['3872-'+instance]>=40, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 7
reduced_data[plco_field] = reduced_data[plco_field] / 7

# How many live births.
# PLCO -- Biobank
# 0 (0) -- 
# 1 (1) -- 
# 2 (2) --
# 3 (3) --
# 4 (4) --
# 5 (5+) --
plco_field = 'livec'
reduced_data[plco_field] = np.nan
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = np.logical_and(raw_data['2734-'+instance]==0, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 0  
    idx = np.logical_and(raw_data['2734-'+instance]==1, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(raw_data['2734-'+instance]==2, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 2  
    idx = np.logical_and(raw_data['2734-'+instance]==3, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 3
    idx = np.logical_and(raw_data['2734-'+instance]==4, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 4  
    idx = np.logical_and(raw_data['2734-'+instance]>=5, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 5
reduced_data[plco_field] = reduced_data[plco_field] / 5

# How many miscarriages or abortion
# PLCO -- Biobank
plco_field = 'miscar'
reduced_data[plco_field] = np.nan
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    field_mis = '3839-'+instance
    field_abort = '3849-'+instance
    total = raw_data[field_mis] + raw_data[field_abort]
    idx = np.logical_and(total==0, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 0  
    idx = np.logical_and(total==1, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(total>=2, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 2  
reduced_data[plco_field] = reduced_data[plco_field] / 2

# Ever been pregnant
# PLCO -- Biobank
# 0 (No) --
# 1 (Yes) -- 
# 2 (Don't Know) --
plco_field = 'preg_f'
reduced_data[plco_field] = np.nan
plco_codes = [0, 1]
ukbb_codes = [[0], [1]]
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    idx = np.logical_and(np.logical_and(raw_data['2774-'+instance]==1, raw_data['2734-'+instance]>0),
                         np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1

# How many still births
# PLCO -- Biobank
plco_field = 'stillb'
reduced_data[plco_field] = np.nan
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    total = raw_data[field_mis] + raw_data[field_abort]
    idx = np.logical_and(raw_data['3829-'+instance]==0, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 0  
    idx = np.logical_and(raw_data['3829-'+instance]==1, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(raw_data['3829-'+instance]>=2, np.isnan(reduced_data[plco_field]))
    reduced_data.loc[idx, plco_field] = 2  
reduced_data[plco_field] = reduced_data[plco_field] / 2

# Trying to get pregnant unsuccessfully
# PLCO -- Biobank
plco_field = 'trypreg'
reduced_data[plco_field] = 0
plco_codes = [1]
ukbb_codes = [['N970','N971','N972','N973','N974','N978','N979']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Tubal pregnancy
# PLCO -- Biobank
plco_field = 'tubal'
reduced_data[plco_field] = 0
plco_codes = [1]
ukbb_codes = [['O001','O002','O008','O009']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
reduced_data[plco_field] = reduced_data[plco_field] / 2

# Menarche
# PLCO -- Biobank
# 0 (Not Applicable) -- 
# 1 (<10) -- 
# 2 (10-12) --
# 3 (12-14) --
# 4 (14-16) --
# 5 (>=16) --
plco_field = 'fmenstr'
reduced_data[plco_field] = 0
instances = ['0.0', '1.0', '2.0', '3.0']
for instance in instances:
    field = '2714-'+instance
    idx = np.logical_and(raw_data[field]<10, reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 1  
    idx = np.logical_and(np.logical_and(raw_data[field]>=10, raw_data[field]<12),
                         reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 2
    idx = np.logical_and(np.logical_and(raw_data[field]>=12, raw_data[field]<14),
                         reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 3
    idx = np.logical_and(np.logical_and(raw_data[field]>=14, raw_data[field]<16),
                         reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 4
    idx = np.logical_and(raw_data[field]>16, reduced_data[plco_field]==0)
    reduced_data.loc[idx, plco_field] = 5
reduced_data[plco_field] = reduced_data[plco_field] / 5

# Benign or Fibrocystic Breast Disease
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'bbd'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['N600', 'N601', 'N602', 'N603', 'N604', 'N608', 'N609']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Benign Ovarian Tumor/Cyst
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'benign_ovcyst'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['D27',
                   'E282'
                   'N830','N832'
                   'Q500']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Endometriosis
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'endometriosis'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['N800','N801','N802','N803','N804','N805','N806','N808','N809']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Uterine Fibroid Tumors
# I cannot find this in UKBiobank
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'uterine_fib'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['D250','D251','D252','D259']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

print('Male specific questions')
# Enlarged Prostate
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'enlpros_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['N40','N400']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            
# Age diagnosised with enlarged prostate
# float
plco_field = 'enlprosa'
reduced_data[plco_field] = 0
surg_date = np.array([NA_date] * len(reduced_data[plco_field]))
ukbb_codes = [['N40','N400']]
for instance in range(cond_len):
    for old_code in ukbb_codes[0]:
        idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, surg_date==NA_date)
        surg_date[idx] = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in raw_data.loc[idx, '41280-0.'+str(instance)]]
reduced_data[plco_field] = surg_date
reduced_data[plco_field] = (reduced_data[plco_field] - birth_date).dt.days/365.24
reduced_data[plco_field] = reduced_data[plco_field] / 6
idx = reduced_data[plco_field] < 0
reduced_data.loc[idx, plco_field] = np.nan

# Inflamed Prostate
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'infpros_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['N410','N411','N412','N413','N418','N419']]
for instance in range(cond_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code
            
# Age diagnosised with inflamed prostate
# PLCO -- Biobank
# 0 (Not Applicable) -- 
# 1 (<30) -- 
# 2 (30-40) -- 
# 3 (40-50) --
# 4 (50-60) --
# 5 (60-70) --
# 6 (>=70) --
plco_field = 'infprosa'
reduced_data[plco_field] = 0
surg_date = np.array([NA_date] * len(reduced_data[plco_field]))
ukbb_codes = [['N410','N411','N412','N413','N418','N419']]
for instance in range(cond_len):
    for old_code in ukbb_codes[0]:
        idx = np.logical_and(raw_data['41270-0.'+str(instance)]==old_code, surg_date==NA_date)
        surg_date[idx] = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in raw_data.loc[idx, '41280-0.'+str(instance)]]
surg_age = np.zeros(len(surg_date))
for i in range(len(surg_age)):
    surg_age[i] = (surg_date[i] - birth_date[i]).days/365.25
idx = np.logical_and(surg_age>0, np.logical_and(surg_age<30, reduced_data[plco_field]==0))
reduced_data.loc[idx, plco_field] = 1  
idx = np.logical_and(np.logical_and(surg_age>=30, surg_age<40),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 2
idx = np.logical_and(np.logical_and(surg_age>=40, surg_age<50),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 3
idx = np.logical_and(np.logical_and(surg_age>=50, surg_age<60),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 4
idx = np.logical_and(np.logical_and(surg_age>=60, surg_age<70),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 5
idx = np.logical_and(surg_age>=70, reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 6
reduced_data[plco_field] = reduced_data[plco_field] / 6

# How often do you urinate at night
# I cannot find this in UKBiobank
plco_field = 'urinate_f'
reduced_data[plco_field] = np.nan

# Age when you began urinating at night
# I cannot find this in UKBiobank
plco_field = 'urinatea'
reduced_data[plco_field] = np.nan

# Age at prostate surgery
# PLCO -- Biobank
# 0 (Not Applicable) -- 
# 1 (<40) -- 
# 2 (40-50) --
# 3 (50-60) --
# 4 (60-70) --
# 5 (>=70) --
plco_field = 'surg_age'
reduced_data[plco_field] = 0
surg_date = np.array([NA_date] * len(reduced_data[plco_field]))
ukbb_codes = [['M452', 'M454',
               'M611', 'M612', 'M613', 'M614', 'M618', 'M619',
               'M621', 'M622', 'M623', 'M624', 'M628', 'M629',
               'M651','M652','M653','M654','M655','M656',
               'M671','M672','M673','M674','M675','M676','M678','M679',
               'M681','M682','M688','M689','M683',
               'M701','M702','M703','M704','M705','M706','M707',
               'M711','M712','M718','M719',
               'Z387','Z422']]
for instance in range(surg_len):
    for old_code in ukbb_codes[0]:
        idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, surg_date==NA_date)
        surg_date[idx] = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in raw_data.loc[idx, '41282-0.'+str(instance)]]
surg_age = np.zeros(len(surg_date))
for i in range(len(surg_age)):
    surg_age[i] = (surg_date[i] - birth_date[i]).days/365.25
idx = np.logical_and(surg_age>0, np.logical_and(surg_age<40, reduced_data[plco_field]==0))
reduced_data.loc[idx, plco_field] = 1  
idx = np.logical_and(np.logical_and(surg_age>=40, surg_age<50),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 2
idx = np.logical_and(np.logical_and(surg_age>=50, surg_age<60),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 3
idx = np.logical_and(np.logical_and(surg_age>=60, surg_age<70),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 4
idx = np.logical_and(surg_age>=70, reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 5
reduced_data[plco_field] = reduced_data[plco_field] / 5

# Ever had Biopsy of Prostate
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'surg_biopsy'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['M452', 'M454',
                   'M701','M702','M703']]
for instance in range(surg_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Prostatectomy for Benign Disease
# I do not think I can get this surgery for BENIGN disease specifically
plco_field = 'surg_prostatectomy'
reduced_data[plco_field] = np.nan

# Ever had Transurethral Resection of Prostate
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'surg_resection'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[],'M651','M652','M653','M654','M655','M656']
for instance in range(surg_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Ever had vasectomy
# PLCO -- Biobank
# 0 (No) -- 0 (No)
# 1 (Yes) -- 1 (Yes)
plco_field = 'vasect_f'
reduced_data[plco_field] = 0
plco_codes = [0, 1]
ukbb_codes = [[], ['N171','N172','N178','N179']]
for instance in range(surg_len):
    for i, new_code in enumerate(plco_codes):
        for old_code in ukbb_codes[i]:
            idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, reduced_data[plco_field]==0)
            reduced_data.loc[idx, plco_field] = new_code

# Age at vasectomy
# PLCO -- Biobank
# 0 (Not Applicable) -- 
# 1 (<25) -- 
# 2 (25-35) --
# 3 (35-40) --
# 4 (>=40) --
plco_field = 'vasecta'
reduced_data[plco_field] = 0
surg_date = np.array([NA_date] * len(reduced_data[plco_field]))
ukbb_codes = [['N171','N172','N178','N179']]
for instance in range(surg_len):
    for old_code in ukbb_codes[0]:
        idx = np.logical_and(raw_data['41272-0.'+str(instance)]==old_code, surg_date==NA_date)
        surg_date[idx] = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in raw_data.loc[idx, '41282-0.'+str(instance)]]
surg_age = np.zeros(len(surg_date))
for i in range(len(surg_age)):
    surg_age[i] = (surg_date[i] - birth_date[i]).days/365.25
idx = np.logical_and(surg_age>0, np.logical_and(surg_age<25, reduced_data[plco_field]==0))
reduced_data.loc[idx, plco_field] = 1  
idx = np.logical_and(np.logical_and(surg_age>=25, surg_age<35),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 2
idx = np.logical_and(np.logical_and(surg_age>=35, surg_age<45),
                     reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 3
idx = np.logical_and(surg_age>=45, reduced_data[plco_field]==0)
reduced_data.loc[idx, plco_field] = 4
reduced_data[plco_field] = reduced_data[plco_field] / 4

# # Add missing indicators
# col_names = reduced_data.columns
# for name in col_names:
#     reduced_data["I"+name] = ~np.isnan(reduced_data[name])
    
reduced_data.to_csv('finalStatBioUKB.csv', index=False)