import pandas as pd
from scipy.stats import gmean

pangu = pd.read_csv("pangu2.csv")
superlu = pd.read_csv("superlu2.csv")


pangu['Others-pre'] = pangu['Numeric factorization time-pre']
pangu['GETRF'] = pangu['GETRF']+pangu['SSSSM']+pangu['GESSM']+pangu['TSTRF']
pangu['TSTRF'] = pangu['TSTRF']+pangu['SSSSM']+pangu['GESSM']
pangu['GESSM'] = pangu['GESSM']+pangu['SSSSM']
pangu['SSSSM'] = pangu['SSSSM']

pangu['Others-after'] = pangu['Numeric factorization time-after']

superlu['SuperLU910_kernel_after_add'] = superlu['SuperLU910_kernel']+superlu['SuperLU910_scatter']


pangu.to_csv("pangu2-1.csv", index=False)
superlu.to_csv("superlu2-1.csv", index=False)

data_4090 = pd.read_csv("pangu2-1.csv")

if 'matrix' in data_4090.columns:
    columns_to_calculate = data_4090.columns.drop('matrix')
else:
    columns_to_calculate = data_4090.columns

geometric_means = gmean(data_4090[columns_to_calculate], axis=0)

new_row = pd.Series([float('nan')] * len(data_4090.columns), index=data_4090.columns)
new_row[columns_to_calculate] = geometric_means

if 'matrix' in data_4090.columns:
    new_row['matrix'] = 'Geometric Mean'
else:
    print("Geometric Mean,", end='')

data_4090 = pd.concat([data_4090, new_row.to_frame().T], ignore_index=True)

data_4090.to_csv("pangu2-1.csv", index=False)

for col in columns_to_calculate:
    print(f"{new_row[col]:.6f}", end=',')


data_4090 = pd.read_csv("superlu2-1.csv")

if 'matrix' in data_4090.columns:
    columns_to_calculate = data_4090.columns.drop('matrix')
else:
    columns_to_calculate = data_4090.columns

geometric_means = gmean(data_4090[columns_to_calculate], axis=0)

new_row = pd.Series([float('nan')] * len(data_4090.columns), index=data_4090.columns)
new_row[columns_to_calculate] = geometric_means

if 'matrix' in data_4090.columns:
    new_row['matrix'] = 'Geometric Mean'
else:
    print("Geometric Mean,", end='')

data_4090 = pd.concat([data_4090, new_row.to_frame().T], ignore_index=True)

data_4090.to_csv("superlu2-1.csv", index=False)

for col in columns_to_calculate:
    print(f"{new_row[col]:.6f}", end=',')