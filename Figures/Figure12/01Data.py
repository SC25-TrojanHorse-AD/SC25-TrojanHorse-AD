import pandas as pd

csv_files = ['pangu', 'pangu_sc', 'superlu', 'superlu_sc']
for csv_name in csv_files:
    data = pd.read_csv("./GFlops/" + csv_name + ".csv")
    
    if csv_name in ['pangu', 'pangu_sc']:
        flop_data = pd.read_csv("./GFlops/PanguLU_flop.csv")
    elif csv_name in ['superlu', 'superlu_sc']:
        flop_data = pd.read_csv("./GFlops/SuperLU_flop.csv")
    
    data['flops'] = flop_data['flops']
    
    num_list = ['1', '2', '4', '8', '16', '32']
    for num in num_list:
        data['GFflops_' + str(num)] = data['flops'] / 1e9 / data[num]
    
    data = data.T.reset_index()
    data.columns = data.iloc[0]
    data = data.drop(0)
    data.to_csv("./GFlops/" + csv_name + "_withGFlops.csv", index=False)