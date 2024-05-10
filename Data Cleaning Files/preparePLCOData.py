import os
import numpy as np
import pandas as pd

fileName = 'Data Files/lung.data.feb16.d080516.csv' 

def readData():
    PLCOData = pd.read_csv(fileName)
    return PLCOData

def processData(fullPLCOData):
    PLCOData = pd.DataFrame()
    
    # Remove nonresponders and those that have already had cancer
    fullPLCOData = fullPLCOData.loc[fullPLCOData['bq_returned']!=0,:]
    #fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysc']!='D',:]
    #fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysc']!='M',:]
    fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysl']!='D',:]
    fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysl']!='M',:]
    #fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysp']!='D',:]
    #fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysp']!='M',:]
    #fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysosumm']!='D',:]
    #fullPLCOData = fullPLCOData.loc[fullPLCOData['candxdaysosumm']!='M',:]
    cancers = ['lung']
    #for cancer in cancers:
        #fullPLCOData = fullPLCOData.loc[fullPLCOData[cancer + '_candxdays']!='D',:]
    
    
    ## Demograpics (6)
    # Set Gender with 0=Male and 1=Female
    PLCOData['gender'] = fullPLCOData['sex'] - 1
    
    # Normalize age
    PLCOData['age'] = fullPLCOData['bq_age'].astype('float')/100
    
    # One-hot encode race
    fullPLCOData['race7'] = fullPLCOData['race7'].astype(int)
    PLCOData['raceWh']  = 1 * (fullPLCOData['race7']==1)
    PLCOData['raceBl']  = 1 * (fullPLCOData['race7']==2)
    PLCOData['raceHis'] = 1 * (fullPLCOData['race7']==3)
    PLCOData['raceAs']  = 1 * (fullPLCOData['race7']==4)
    PLCOData['racePI']  = 1 * (fullPLCOData['race7']==5)
    PLCOData['raceAI']  = 1 * (fullPLCOData['race7']==6)
    PLCOData['raceMis'] = 1 * (fullPLCOData['race7']==7)
    
    # One-hot  encode education
    fullPLCOData.loc[fullPLCOData['educat']=='M', 'educat'] = np.nan
    fullPLCOData['educat'] = fullPLCOData['educat'].astype({'educat': float})
    PLCOData['edlt8']  = 1 * (fullPLCOData['educat']==1)
    PLCOData['ed8t11'] = 1 * (fullPLCOData['educat']==2)
    PLCOData['ed12']   = 1 * (fullPLCOData['educat']==3)
    PLCOData['edphs']  = 1 * (fullPLCOData['educat']==4)
    PLCOData['edscol'] = 1 * (fullPLCOData['educat']==5)
    PLCOData['edcol']  = 1 * (fullPLCOData['educat']==6)
    PLCOData['edpcol'] = 1 * (fullPLCOData['educat']==7)
    
    # One-hot encode marital status
    fullPLCOData.loc[fullPLCOData['marital']=='M', 'marital'] = np.nan
    fullPLCOData['marital'] = fullPLCOData['marital'].astype({'marital': float})
    PLCOData['marmar'] = 1 * (fullPLCOData['marital']==1)
    PLCOData['marwid'] = 1 * (fullPLCOData['marital']==2)
    PLCOData['mardiv'] = 1 * (fullPLCOData['marital']==3)
    PLCOData['marsep'] = 1 * (fullPLCOData['marital']==4)
    PLCOData['marnvm'] = 1 * (fullPLCOData['marital']==5)
    
    # One-hot encode occupation status
    fullPLCOData.loc[fullPLCOData['occupat']=='M', 'occupat'] = np.nan
    fullPLCOData['occupat'] = fullPLCOData['occupat'].astype({'occupat': float})
    PLCOData['occhmk'] = 1 * (fullPLCOData['occupat']==1)
    PLCOData['occwrk'] = 1 * (fullPLCOData['occupat']==2)
    PLCOData['occune'] = 1 * (fullPLCOData['occupat']==3)
    PLCOData['occret'] = 1 * (fullPLCOData['occupat']==4)
    PLCOData['occesl'] = 1 * (fullPLCOData['occupat']==5)
    PLCOData['occdis'] = 1 * (fullPLCOData['occupat']==6)
    PLCOData['occoth'] = 1 * (fullPLCOData['occupat']==7)
    
    ## Body type (4)
    # Normalize current BMI
    fullPLCOData.loc[fullPLCOData['bmi_curr']=='M', 'bmi_curr'] = np.nan
    fullPLCOData.loc[fullPLCOData['bmi_curr']=='R', 'bmi_curr'] = np.nan
    PLCOData['bmi_curr'] = fullPLCOData['bmi_curr'].astype(float)/100
    
#    # Normalize current weight
#    fullPLCOData.loc[fullPLCOData['weight_f']=='M', 'weight_f'] = np.nan
#    fullPLCOData.loc[fullPLCOData['weight_f']=='R', 'weight_f'] = np.nan
#    PLCOData['weight_f'] = fullPLCOData['weight_f'].astype(float)/500
#    
#    # Normalize weight at 50 years old
#    fullPLCOData.loc[fullPLCOData['weight50_f']=='M', 'weight50_f'] = np.nan
#    fullPLCOData.loc[fullPLCOData['weight50_f']=='R', 'weight50_f'] = np.nan
#    PLCOData['weight50_f'] = fullPLCOData['weight50_f'].astype(float)/500
    
    # Normalize weight at 20 years old
    fullPLCOData.loc[fullPLCOData['weight20_f']=='M', 'weight20_f'] = np.nan
    fullPLCOData.loc[fullPLCOData['weight20_f']=='R', 'weight20_f'] = np.nan
    PLCOData['weight20_f'] = fullPLCOData['weight20_f'].astype(float)/500
    
    ## Diseases (13)
    fullPLCOData.loc[fullPLCOData['arthrit_f']   =='M', 'arthrit_f']         = np.nan
    fullPLCOData.loc[fullPLCOData['bronchit_f']  =='M', 'bronchit_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['colon_comorbidity']=='M', 'colon_comorbidity'] = np.nan
    fullPLCOData.loc[fullPLCOData['diabetes_f']  =='M', 'diabetes_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['divertic_f']  =='M', 'divertic_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['emphys_f']    =='M', 'emphys_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['gallblad_f']  =='M', 'gallblad_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['hearta_f']    =='M', 'hearta_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['hyperten_f']  =='M', 'hyperten_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['liver_comorbidity']=='M', 'liver_comorbidity'] = np.nan
    fullPLCOData.loc[fullPLCOData['osteopor_f']  =='M', 'osteopor_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['polyps_f']    =='M', 'polyps_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['stroke_f']    =='M', 'stroke_f']          = np.nan
    PLCOData['arthrit_f']    = fullPLCOData['arthrit_f'].astype(float)/1
    PLCOData['bronchit_f']   = fullPLCOData['bronchit_f'].astype(float)/1
    PLCOData['colon_comorb'] = fullPLCOData['colon_comorbidity'].astype(float)/1
    PLCOData['diabetes_f']   = fullPLCOData['diabetes_f'].astype(float)/1
    PLCOData['divertic_f']   = fullPLCOData['divertic_f'].astype(float)/1
    PLCOData['emphys_f']     = fullPLCOData['emphys_f'].astype(float)/1
    PLCOData['gallblad_f']   = fullPLCOData['gallblad_f'].astype(float)/1
    PLCOData['hearta_f']     = fullPLCOData['hearta_f'].astype(float)/1
    PLCOData['hyperten_f']   = fullPLCOData['hyperten_f'].astype(float)/1
    PLCOData['liver_comorb'] = fullPLCOData['liver_comorbidity'].astype(float)/1
    PLCOData['osteopor_f']   = fullPLCOData['osteopor_f'].astype(float)/1
    PLCOData['polyps_f']     = fullPLCOData['polyps_f'].astype(float)/1
    PLCOData['stroke_f']     = fullPLCOData['stroke_f'].astype(float)/1
    
    ## Smoking (4)
    fullPLCOData.loc[fullPLCOData['smoked_f']  =='M', 'smoked_f']   = np.nan
    fullPLCOData.loc[fullPLCOData['cig_stat']  =='M', 'cig_stat']   = np.nan
    fullPLCOData.loc[fullPLCOData['cig_stat']  =='A', 'cig_stat']   = np.nan
    fullPLCOData.loc[fullPLCOData['cig_stop']  =='M', 'cig_stop']   = np.nan
    fullPLCOData.loc[fullPLCOData['cig_stop']  =='N', 'cig_stop']   = np.nan
    fullPLCOData.loc[fullPLCOData['pack_years']=='M', 'pack_years'] = np.nan
    PLCOData['SMKEV']  = fullPLCOData['smoked_f'].astype(float)
    PLCOData['SMKNOW'] = fullPLCOData['cig_stat'].astype(float)
    PLCOData.loc[PLCOData['SMKNOW']==2,'SMKNOW'] = 0
    PLCOData['cig_stop']   = fullPLCOData['cig_stop'].astype(float)/100
    PLCOData['pack_years'] = fullPLCOData['pack_years'].astype(float)/500
    
    ## NSAIDS (2)
    fullPLCOData.loc[fullPLCOData['asppd']  =='M', 'asppd']  = np.nan
    fullPLCOData.loc[fullPLCOData['ibuppd'] =='M', 'ibuppd'] = np.nan
    PLCOData['asppd']  = fullPLCOData['asppd'].astype(float)
    PLCOData['ibuppd'] = fullPLCOData['ibuppd'].astype(float)
    PLCOData.loc[PLCOData['asppd'] ==1, 'asppd']  = 30
    PLCOData.loc[PLCOData['asppd'] ==2, 'asppd']  = 30*2
    PLCOData.loc[PLCOData['asppd'] ==3, 'asppd']  = 4
    PLCOData.loc[PLCOData['asppd'] ==4, 'asppd']  = 4*2
    PLCOData.loc[PLCOData['asppd'] ==5, 'asppd']  = 4*3
    PLCOData.loc[PLCOData['asppd'] ==6, 'asppd']  = 1
    PLCOData.loc[PLCOData['asppd'] ==7, 'asppd']  = 2
    PLCOData.loc[PLCOData['ibuppd']==1, 'ibuppd'] = 30
    PLCOData.loc[PLCOData['ibuppd']==2, 'ibuppd'] = 30*2
    PLCOData.loc[PLCOData['ibuppd']==3, 'ibuppd'] = 4
    PLCOData.loc[PLCOData['ibuppd']==4, 'ibuppd'] = 4*2
    PLCOData.loc[PLCOData['ibuppd']==5, 'ibuppd'] = 4*3
    PLCOData.loc[PLCOData['ibuppd']==6, 'ibuppd'] = 1
    PLCOData.loc[PLCOData['ibuppd']==7, 'ibuppd'] = 2
    PLCOData['asppd']  = PLCOData['asppd']/60
    PLCOData['ibuppd'] = PLCOData['ibuppd']/60
    
    ## Family History (3*18)
    cancers = ['lung']
    for cancer in cancers:
        fullPLCOData.loc[fullPLCOData[cancer + '_fh']     =='M', cancer + '_fh']     = np.nan
        fullPLCOData.loc[fullPLCOData[cancer + '_fh']     =='9', cancer + '_fh']     = np.nan
        fullPLCOData.loc[fullPLCOData[cancer + '_fh_age'] =='M', cancer + '_fh_age'] = np.nan
        fullPLCOData.loc[fullPLCOData[cancer + '_fh_age'] =='N', cancer + '_fh_age'] = np.nan
        fullPLCOData.loc[fullPLCOData[cancer + '_fh_age'] =='A', cancer + '_fh_age'] = np.nan
        fullPLCOData.loc[fullPLCOData[cancer + '_fh_cnt'] =='M', cancer + '_fh_cnt'] = np.nan
        PLCOData[cancer + '_fh']     = fullPLCOData[cancer + '_fh'].astype(float)
        PLCOData[cancer + '_fh_age'] = fullPLCOData[cancer + '_fh_age'].astype(float)/100
        PLCOData[cancer + '_fh_cnt'] = fullPLCOData[cancer + '_fh_cnt'].astype(float)/10
        
    ## Previous Cancer (2*18)
    cancers = ['lung']
    for cancer in cancers:
        PLCOData['trial_ph_' + cancer] = fullPLCOData['trial_ph_' + cancer].astype(float)
        PLCOData.loc[PLCOData['trial_ph_' + cancer]==9, 'trial_ph_' + cancer] = np.nan
        PLCOData[cancer + '_is_first_dx'] = fullPLCOData[cancer + '_is_first_dx'].astype(float)
    
    ## Female Specific (20)
    fullPLCOData.loc[fullPLCOData['hystera']      =='M', 'hystera']       = np.nan
    fullPLCOData.loc[fullPLCOData['hystera']      =='N', 'hystera']       = np.nan
    fullPLCOData.loc[fullPLCOData['hystera']      =='G', 'hystera']       = np.nan
    fullPLCOData.loc[fullPLCOData['ovariesr_f']   =='M', 'ovariesr_f']    = np.nan
    fullPLCOData.loc[fullPLCOData['ovariesr_f']   =='G', 'ovariesr_f']    = np.nan
    fullPLCOData.loc[fullPLCOData['ovariesr_f']   =='5', 'ovariesr_f']    = np.nan
    fullPLCOData.loc[fullPLCOData['ovariesr_f']   =='8', 'ovariesr_f']    = np.nan
    fullPLCOData.loc[fullPLCOData['tuballig']     =='M', 'tuballig']      = np.nan
    fullPLCOData.loc[fullPLCOData['tuballig']     =='G', 'tuballig']      = np.nan
    fullPLCOData.loc[fullPLCOData['tuballig']     =='2', 'tuballig']      = np.nan
    fullPLCOData.loc[fullPLCOData['bcontr_f']     =='M', 'bcontr_f']      = np.nan
    fullPLCOData.loc[fullPLCOData['bcontr_f']     =='G', 'bcontr_f']      = np.nan
    fullPLCOData.loc[fullPLCOData['bcontra']      =='M', 'bcontra']       = np.nan
    fullPLCOData.loc[fullPLCOData['bcontra']      =='N', 'bcontra']       = np.nan
    fullPLCOData.loc[fullPLCOData['bcontra']      =='G', 'bcontra']       = np.nan
    fullPLCOData.loc[fullPLCOData['curhorm']      =='M', 'curhorm']       = np.nan
    fullPLCOData.loc[fullPLCOData['curhorm']      =='G', 'curhorm']       = np.nan
    fullPLCOData.loc[fullPLCOData['horm_f']       =='M', 'horm_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['horm_f']       =='G', 'horm_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['horm_f']       =='2', 'horm_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['thorm']        =='M', 'thorm']         = np.nan
    fullPLCOData.loc[fullPLCOData['thorm']        =='G', 'thorm']         = np.nan
    fullPLCOData.loc[fullPLCOData['fchilda']      =='M', 'fchilda']       = np.nan
    fullPLCOData.loc[fullPLCOData['fchilda']      =='N', 'fchilda']       = np.nan
    fullPLCOData.loc[fullPLCOData['fchilda']      =='G', 'fchilda']       = np.nan
    fullPLCOData.loc[fullPLCOData['livec']        =='M', 'livec']         = np.nan
    fullPLCOData.loc[fullPLCOData['livec']        =='G', 'livec']         = np.nan
    fullPLCOData.loc[fullPLCOData['miscar']       =='M', 'miscar']        = np.nan
    fullPLCOData.loc[fullPLCOData['miscar']       =='G', 'miscar']        = np.nan
    fullPLCOData.loc[fullPLCOData['preg_f']       =='M', 'preg_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['preg_f']       =='G', 'preg_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['preg_f']       =='2', 'preg_f']        = np.nan
    fullPLCOData.loc[fullPLCOData['stillb']       =='M', 'stillb']        = np.nan
    fullPLCOData.loc[fullPLCOData['stillb']       =='G', 'stillb']        = np.nan
    fullPLCOData.loc[fullPLCOData['trypreg']      =='M', 'trypreg']       = np.nan
    fullPLCOData.loc[fullPLCOData['trypreg']      =='G', 'trypreg']       = np.nan
    fullPLCOData.loc[fullPLCOData['tubal']        =='M', 'tubal']         = np.nan
    fullPLCOData.loc[fullPLCOData['tubal']        =='G', 'tubal']         = np.nan
    fullPLCOData.loc[fullPLCOData['fmenstr']      =='M', 'fmenstr']       = np.nan
    fullPLCOData.loc[fullPLCOData['fmenstr']      =='G', 'fmenstr']       = np.nan
    fullPLCOData.loc[fullPLCOData['bbd']          =='M', 'bbd']           = np.nan
    fullPLCOData.loc[fullPLCOData['bbd']          =='G', 'bbd']           = np.nan
    fullPLCOData.loc[fullPLCOData['benign_ovcyst']=='M', 'benign_ovcyst'] = np.nan
    fullPLCOData.loc[fullPLCOData['benign_ovcyst']=='G', 'benign_ovcyst'] = np.nan
    fullPLCOData.loc[fullPLCOData['endometriosis']=='M', 'endometriosis'] = np.nan
    fullPLCOData.loc[fullPLCOData['endometriosis']=='G', 'endometriosis'] = np.nan
    fullPLCOData.loc[fullPLCOData['uterine_fib']  =='M', 'uterine_fib']   = np.nan
    fullPLCOData.loc[fullPLCOData['uterine_fib']  =='G', 'uterine_fib']   = np.nan
    PLCOData['hystera']       = fullPLCOData['hystera'].astype(float)/5
    PLCOData['ovariesr_f']    = fullPLCOData['ovariesr_f'].astype(float)/4
    PLCOData['tuballig']      = fullPLCOData['tuballig'].astype(float)
    PLCOData['bcontr_f']      = fullPLCOData['bcontr_f'].astype(float)
    PLCOData['bcontra']       = fullPLCOData['bcontra'].astype(float)/4
    PLCOData['curhorm']       = fullPLCOData['curhorm'].astype(float)
    PLCOData['horm_f']        = fullPLCOData['horm_f'].astype(float)
    PLCOData['thorm']         = fullPLCOData['thorm'].astype(float)/5
    PLCOData['fchilda']       = fullPLCOData['fchilda'].astype(float)/7
    PLCOData['livec']         = fullPLCOData['livec'].astype(float)/5
    PLCOData['miscar']        = fullPLCOData['miscar'].astype(float)/2
    PLCOData['preg_f']        = fullPLCOData['preg_f'].astype(float)
    PLCOData['stillb']        = fullPLCOData['stillb'].astype(float)/2
    PLCOData['trypreg']       = fullPLCOData['trypreg'].astype(float)
    PLCOData['tubal']         = fullPLCOData['tubal'].astype(float)/2
    PLCOData['fmenstr']       = fullPLCOData['fmenstr'].astype(float)/5
    PLCOData['bbd']           = fullPLCOData['bbd'].astype(float)
    PLCOData['benign_ovcyst'] = fullPLCOData['benign_ovcyst'].astype(float)
    PLCOData['endometriosis'] = fullPLCOData['endometriosis'].astype(float)
    PLCOData['uterine_fib']   = fullPLCOData['uterine_fib'].astype(float)
    
    ## Male Specific (12)
    fullPLCOData.loc[fullPLCOData['enlpros_f']         =='M', 'enlpros_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['enlpros_f']         =='G', 'enlpros_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['enlprosa']          =='M', 'enlprosa']           = np.nan
    fullPLCOData.loc[fullPLCOData['enlprosa']          =='G', 'enlprosa']           = np.nan
    fullPLCOData.loc[fullPLCOData['enlprosa']          =='N', 'enlprosa']           = np.nan
    fullPLCOData.loc[fullPLCOData['infpros_f']         =='M', 'infpros_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['infpros_f']         =='G', 'infpros_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['infprosa']          =='M', 'infprosa']           = np.nan
    fullPLCOData.loc[fullPLCOData['infprosa']          =='G', 'infprosa']           = np.nan
    fullPLCOData.loc[fullPLCOData['infprosa']          =='N', 'infprosa']           = np.nan
    fullPLCOData.loc[fullPLCOData['urinate_f']         =='M', 'urinate_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['urinate_f']         =='G', 'urinate_f']          = np.nan
    fullPLCOData.loc[fullPLCOData['urinatea']          =='M', 'urinatea']           = np.nan
    fullPLCOData.loc[fullPLCOData['urinatea']          =='G', 'urinatea']           = np.nan
    fullPLCOData.loc[fullPLCOData['urinatea']          =='N', 'urinatea']           = np.nan
    fullPLCOData.loc[fullPLCOData['surg_age']          =='M', 'surg_age']           = np.nan
    fullPLCOData.loc[fullPLCOData['surg_age']          =='G', 'surg_age']           = np.nan
    fullPLCOData.loc[fullPLCOData['surg_age']          =='N', 'surg_age']           = np.nan
    fullPLCOData.loc[fullPLCOData['surg_biopsy']       =='M', 'surg_biopsy']        = np.nan
    fullPLCOData.loc[fullPLCOData['surg_biopsy']       =='G', 'surg_biopsy']        = np.nan
    fullPLCOData.loc[fullPLCOData['surg_prostatectomy']=='M', 'surg_prostatectomy'] = np.nan
    fullPLCOData.loc[fullPLCOData['surg_prostatectomy']=='G', 'surg_prostatectomy'] = np.nan
    fullPLCOData.loc[fullPLCOData['surg_resection']    =='M', 'surg_resection']     = np.nan
    fullPLCOData.loc[fullPLCOData['surg_resection']    =='G', 'surg_resection']     = np.nan
    fullPLCOData.loc[fullPLCOData['vasect_f']          =='M', 'vasect_f']           = np.nan
    fullPLCOData.loc[fullPLCOData['vasect_f']          =='G', 'vasect_f']           = np.nan
    fullPLCOData.loc[fullPLCOData['vasecta']           =='M', 'vasecta']            = np.nan
    fullPLCOData.loc[fullPLCOData['vasecta']           =='G', 'vasecta']            = np.nan
    fullPLCOData.loc[fullPLCOData['vasecta']           =='N', 'vasecta']            = np.nan
    PLCOData['enlpros_f']          = fullPLCOData['enlpros_f'].astype(float)
    PLCOData['enlprosa']           = fullPLCOData['enlprosa'].astype(float)/6
    PLCOData['infpros_f']          = fullPLCOData['infpros_f'].astype(float)
    PLCOData['infprosa']           = fullPLCOData['infprosa'].astype(float)/6
    PLCOData['urinate_f']          = fullPLCOData['urinate_f'].astype(float)/5
    PLCOData['urinatea']           = fullPLCOData['urinatea'].astype(float)/6
    PLCOData['surg_age']           = fullPLCOData['surg_age'].astype(float)/5
    PLCOData['surg_biopsy']        = fullPLCOData['surg_biopsy'].astype(float)
    PLCOData['surg_prostatectomy'] = fullPLCOData['surg_prostatectomy'].astype(float)
    PLCOData['surg_resection']     = fullPLCOData['surg_resection'].astype(float)
    PLCOData['vasect_f']           = fullPLCOData['vasect_f'].astype(float)
    PLCOData['vasecta']            = fullPLCOData['vasecta'].astype(float)/4
    
    ## Cancer Status (18)
    #fullPLCOData.loc[fullPLCOData['candxdaysc']    =='N', 'candxdaysc']     = np.nan
    fullPLCOData.loc[fullPLCOData['candxdaysl']    =='N', 'candxdaysl']     = np.nan
    #fullPLCOData.loc[fullPLCOData['candxdaysp']    =='N', 'candxdaysp']     = np.nan
    #fullPLCOData.loc[fullPLCOData['candxdaysosumm']=='N', 'candxdaysosumm'] = np.nan
    #PLCOData['colo'] = fullPLCOData['candxdaysc'].astype(float)
    PLCOData['lung'] = fullPLCOData['candxdaysl'].astype(float)
    #PLCOData['pros'] = fullPLCOData['candxdaysp'].astype(float)
    #PLCOData['ovar'] = fullPLCOData['candxdaysosumm'].astype(float)
    cancers = ['lung']
    #for cancer in cancers:
       # fullPLCOData.loc[fullPLCOData[cancer + '_candxdays']=='N', cancer + '_candxdays'] = np.nan
        #PLCOData[cancer] = fullPLCOData[cancer + '_candxdays'].astype(float)
    
    return PLCOData

def writeDataFiles(data, days_cutoff):
    male_only_columns = [
        'enlpros_f', 'enlprosa', 'infpros_f', 'infprosa', 'urinate_f', 'urinatea', 
        'surg_age', 'surg_biopsy', 'surg_prostatectomy', 'surg_resection', 
        'vasect_f', 'vasecta', 'mbreast', 'pros'
    ]
    female_only_columns = [
        'hystera', 'ovariesr_f', 'tuballig', 'bcontr_f', 'bcontra', 'curhorm', 
        'horm_f', 'thorm', 'fchilda', 'livec', 'miscar', 'preg_f', 'stillb', 
        'trypreg', 'tubal', 'fmenstr', 'bbd', 'benign_ovcyst', 'endometriosis', 
        'uterine_fib', 'breast', 'endo', 'ovar'
    ]
    male_cancers = ['lung']
    female_cancers = ['lung']
    columns = data.columns

    # Convert the lung cancer column to 0 or 1 based on the cutoff, both for male and female
    data['lung'] = (data['lung'] <= days_cutoff).astype(int)

    # Filter and write male data
    male_data = data.loc[data['gender'] == 0, columns[~columns.isin(female_only_columns)]]
    male_data.to_csv('Male_Lung_Data.csv', index=False)

    # Filter and write female data
    female_data = data.loc[data['gender'] == 1, columns[~columns.isin(male_only_columns)]]
    female_data.to_csv('Female_Lung_Data.csv', index=False)
    
if __name__ == '__main__':
    fullPLCOData = readData()
    PLCOData = processData(fullPLCOData)
    writeDataFiles(PLCOData, 5*365)