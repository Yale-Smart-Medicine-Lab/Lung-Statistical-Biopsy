import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

fileNames = ['Data Files/male_filtered_70_NHIS', 'Data Files/female_filtered_70_NHIS', 
             'Data Files/female_filtered_30_NHIS', 'Data Files/male_filtered_30_NHIS', 
             'Data Files/female_filtered_chosen_NHIS', 'Data Files/male_filtered_chosen_NHIS']
suffix = '_mean_imputed'
ext = '.csv'

for fileName in fileNames:
    data = pd.read_csv(fileName + ext)
    columns_to_keep = []
    for col in data.columns.values:
        # all the values for this feature are not null
        if data[col].notnull().sum() != 0:
            columns_to_keep.append(col)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_data = imputer.fit_transform(data[columns_to_keep])
    imputed_data = pd.DataFrame(imputed_data, columns=columns_to_keep)
    
    imputed_data.to_csv(fileName + suffix + ext, index=False)
    print(f"Imputed data saved to '{fileName + suffix + ext}'.")
