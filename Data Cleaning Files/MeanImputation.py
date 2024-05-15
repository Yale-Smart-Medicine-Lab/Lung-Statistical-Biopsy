import pandas as pd
import numpy as np  # Importing NumPy
from sklearn.impute import SimpleImputer

def impute_csv(file_path, output_file='NHIS_male_chosen_mean_imputed_data.csv'):
    # Load data from a CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Create an imputer object with a mean imputation strategy
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Fit the imputer on the data and transform it
    # This step assumes that all columns of your data are numerical
    imputed_data = imputer.fit_transform(data)

    # Convert the array returned by SimpleImputer back into a pandas DataFrame
    # This uses the original column names and indices
    imputed_df = pd.DataFrame(imputed_data, columns=data.columns, index=data.index)

    # Save the imputed data to a new CSV file
    imputed_df.to_csv(output_file, index=False)
    print(f"Imputed data saved to '{output_file}'.")

# Example usage:
impute_csv('Data Files/male_filtered_chosen_NHIS.csv')
