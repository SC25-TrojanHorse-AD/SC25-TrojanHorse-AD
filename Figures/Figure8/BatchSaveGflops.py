import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


data = pd.read_csv("batch.csv")

plt.scatter(data['time(us)'], data['gflops'], color='b', marker='.', label='Data Points')

plt.title("Gflops vs Time")
plt.xlabel("time(us)")
plt.ylabel("Gflops")



if len(data['time(us)']) < 100:
    step =  int(len(data['time(us)']) / 5)
elif len(data['time(us)']) <500:
    step =  int(len(data['time(us)']) / 20)
elif len(data['time(us)']) < 1000:
    step =  int(len(data['time(us)']) / 25)
elif len(data['time(us)']) < 10000:
    step =  int(len(data['time(us)']) / 200)
else:
    step = 50


max_x_values = []
max_y_values = []
min_x_values = []
min_y_values = []
median_x_values = []  
median_y_values = []  
mean_x_values = []  
mean_y_values = [] 


for i in range(0, len(data['time(us)']), (int)(step/2)):
    subset = data.iloc[i:i + step]
    max_index = subset['gflops'].idxmax()
    min_index = subset['gflops'].idxmin()
    median_index = subset['gflops'].median() 
    median_index = subset['gflops'].sub(median_index).abs().idxmin() 
    mean_value = subset['gflops'].mean() 
    mean_index = subset['gflops'].sub(mean_value).abs().idxmin()  


    max_x_values.append(subset.loc[i, 'time(us)'])
    max_y_values.append(subset.loc[max_index, 'gflops'])
    min_x_values.append(subset.loc[i, 'time(us)'])
    min_y_values.append(subset.loc[min_index, 'gflops'])
    median_x_values.append(subset.loc[i, 'time(us)']) 
    median_y_values.append(subset.loc[median_index, 'gflops'])
    mean_x_values.append(subset.loc[i, 'time(us)']) 
    mean_y_values.append(subset.loc[mean_index, 'gflops'])


plt.plot(max_x_values, max_y_values, color='b', label='Max Points')


plt.plot(min_x_values, min_y_values, color='b', label='Min Points')


plt.plot(median_x_values, median_y_values, color='r', label='Median Points')


plt.plot(mean_x_values, mean_y_values, color='g', label='Mean Points')


plt.fill_between(max_x_values, max_y_values, min_y_values, color='b', alpha=0.2)


plt.legend()



max_data_df = pd.DataFrame({
    'time(us)': max_x_values,
    'gflops': max_y_values
})
max_data_df.to_excel('batchMAX.xlsx', index=False)

min_data_df = pd.DataFrame({
    'time(us)': min_x_values,
    'gflops': min_y_values
})

min_data_df.to_excel('batchMIN.xlsx', index=False)

mean_data_df = pd.DataFrame({
    'time(us)': mean_x_values,
    'gflops': mean_y_values
})
mean_data_df.to_excel('batchMEAN.xlsx', index=False)

print("batch---GFlops---SAVE!!!")