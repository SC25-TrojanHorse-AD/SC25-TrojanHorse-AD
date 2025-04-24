from scipy.stats import gmean
import pandas as pd

data4060 = pd.read_csv('4060.csv')
data4090 = pd.read_csv('4090.csv')
flop_cut = pd.read_csv('flop_cut.csv')

for i in range(len(data4060['matrix'])):
    matrix_name = data4060['matrix'][i]
    matrix_row = data4060[data4060['matrix'] == matrix_name]
    if (~pd.isna(matrix_row['PanguLU_4.2.0_original_4060'])).any():
        time = matrix_row['PanguLU_4.2.0_original_4060'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['PanguLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4060.loc[data4060['matrix'] == matrix_name, 'PanguLU_4.2.0_original_4060'] = gflops
            
    if (~pd.isna(matrix_row['PanguLU_sc_4060'])).any():
        time = matrix_row['PanguLU_sc_4060'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['PanguLU Flop'].values[0]
            gflops = target_flop / time/1e9
            data4060.loc[data4060['matrix'] == matrix_name, 'PanguLU_sc_4060'] = gflops
data4060.to_csv('data4060_with_gflops.csv', index=False)
for i in range(len(data4090['matrix'])):
    matrix_name = data4090['matrix'][i]
    matrix_row = data4090[data4090['matrix'] == matrix_name]
    if (~pd.isna(matrix_row['PanguLU_4.2.0_4090'])).any():
        time = matrix_row['PanguLU_4.2.0_4090'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['PanguLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4090.loc[data4090['matrix'] == matrix_name, 'PanguLU_4.2.0_4090'] = gflops
    if (~pd.isna(matrix_row['PanguLU_sc_4090'])).any():
        time = matrix_row['PanguLU_sc_4090'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['PanguLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4090.loc[data4090['matrix'] == matrix_name, 'PanguLU_sc_4090'] = gflops
data4090.to_csv('data4090_with_gflops.csv', index=False)
for i in range(len(data4060['matrix'])):
    matrix_name = data4060['matrix'][i]
    matrix_row = data4060[data4060['matrix'] == matrix_name]
    if (~pd.isna(matrix_row['superLU_original_4060'])).any():
        time = matrix_row['superLU_original_4060'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['SuperLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4060.loc[data4060['matrix'] == matrix_name, 'superLU_original_4060'] = gflops
    if (~pd.isna(matrix_row['superLU_sc_4060'])).any():
        time = matrix_row['superLU_sc_4060'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['SuperLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4060.loc[data4060['matrix'] == matrix_name, 'superLU_sc_4060'] = gflops
data4060.to_csv('data4060_with_gflops.csv', index=False)
for i in range(len(data4090['matrix'])):
    matrix_name = data4090['matrix'][i]
    matrix_row = data4090[data4090['matrix'] == matrix_name]
    if (~pd.isna(matrix_row['superLU_original_4090'])).any():
        time = matrix_row['superLU_original_4090'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['SuperLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4090.loc[data4090['matrix'] == matrix_name, 'superLU_original_4090'] = gflops
    if (~pd.isna(matrix_row['superLU_sc_4090'])).any():
        time = matrix_row['superLU_sc_4090'].values[0]
        if (matrix_name in flop_cut['matrix'].values):
            target_flop = flop_cut[flop_cut['matrix'] == matrix_name]['SuperLU Flop'].values[0]
            gflops = target_flop / 1e9/time
            data4090.loc[data4090['matrix'] == matrix_name, 'superLU_sc_4090'] = gflops
data4090.to_csv('data4090_with_gflops.csv', index=False)

from scipy.stats import gmean
import pandas as pd

data_4060 = pd.read_csv("data4060_with_gflops.csv")

geometric_mean1 = gmean(data_4060['PanguLU_4.2.0_original_4060'].dropna())
geometric_mean2 = gmean(data_4060['PanguLU_sc_4060'].dropna())
geometric_mean3 = gmean(data_4060['superLU_original_4060'].dropna())
geometric_mean4 = gmean(data_4060['superLU_sc_4060'].dropna())

print(f"Geomatric Mean,{geometric_mean1},{geometric_mean2},{geometric_mean3},{geometric_mean4}")

new_row = {
    'matrix': 'Geometric Mean',
    'PanguLU_4.2.0_original_4060': geometric_mean1,
    'PanguLU_sc_4060': geometric_mean2,
    'superLU_original_4060': geometric_mean3,
    'superLU_sc_4060': geometric_mean4
}

data_4060 = pd.concat([data_4060, pd.DataFrame([new_row])], ignore_index=True)

data_4060.to_csv("data4060_with_gflops.csv", index=False)

data_4090 = pd.read_csv("data4090_with_gflops.csv")
geometric_mean1 = gmean(data_4090['PanguLU_4.2.0_4090'].dropna())
geometric_mean2 = gmean(data_4090['PanguLU_sc_4090'].dropna())
geometric_mean3 = gmean(data_4090['superLU_original_4090'].dropna())
geometric_mean4 = gmean(data_4090['superLU_sc_4090'].dropna())

print(f"Geomatric Mean,{geometric_mean1},{geometric_mean2},{geometric_mean3},{geometric_mean4}")

new_row = {
    'matrix': 'Geometric Mean',
    'PanguLU_4.2.0_4090': geometric_mean1,
    'PanguLU_sc_4090': geometric_mean2,
    'superLU_original_4090': geometric_mean3,
    'superLU_sc_4090': geometric_mean4
}

data_4090 = pd.concat([data_4090, pd.DataFrame([new_row])], ignore_index=True)

data_4090.to_csv("data4090_with_gflops.csv", index=False)