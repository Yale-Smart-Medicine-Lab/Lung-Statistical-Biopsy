import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def impute_data(data_file):
    print('Reading in UK Biobank data')
    data = pd.read_csv(data_file)
    
    # Filter by gender early to ensure you're working with the correct subset of data
    data = data[data['gender'] == 1]
    
    print('Reading in PLCO data')
    plco_data = pd.read_csv('Data Files/female_lung_data_imputed.csv')
    
    # Exclude 'lung' column from PLCO data
    plco_data = plco_data.drop(columns=['lung'])
    
    # Select only the common columns between UK Biobank and PLCO data after gender filtering
    common_columns = data.columns.intersection(plco_data.columns)
    UKBiobankData = data[common_columns]

    if UKBiobankData.empty:
        print('No samples found in the UK Biobank data after filtering.')
        return

    print('Taking care of empty columns')
    # Calculate std on numerical columns only
    num_columns = UKBiobankData.select_dtypes(include=[np.number]).columns
    columnWNoVar = num_columns[UKBiobankData[num_columns].std() == 0].tolist()
    for column in columnWNoVar:
        UKBiobankData[column] = UKBiobankData[column].fillna(UKBiobankData[column].mean())

    print('Beginning imputation')
    imputer = KNNImputer(n_neighbors=5)
    imputed_UKBiobankData = imputer.fit_transform(UKBiobankData)
    
    # When creating the DataFrame, ensure you're using the columns from UKBiobankData
    imputed_UKBiobankData = pd.DataFrame(imputed_UKBiobankData, columns=UKBiobankData.columns)

    print('Saving data to file')
    imputed_UKBiobankData.to_csv('imputed_female_UKBdata.csv', index=False)

# Assuming "Data Files/male_UKBdata.csv" is the correct path to your data file
impute_data("Data Files/female_UKBdata.csv")