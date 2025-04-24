import pandas as pd
import os

batch_file = [f for f in os.listdir('.') if f.endswith('_batch.csv')]

batch = pd.read_csv(batch_file[0])
batch.columns.values[0] = 'time(us)'
batch.columns.values[1] = 'gflops'
batch.columns.values[2] = 'GB/s'
batch['time(us)']  = batch['time(us)']-batch['time(us)'].min()
batch['time(us)']  = batch['time(us)']/1000000
batch.to_csv("batch.csv",index=False)


nobatch_file = [f for f in os.listdir('.') if f.endswith('_nobatch.csv')]

nobatch = pd.read_csv(nobatch_file[0])
nobatch.columns.values[0] = 'time(us)'
nobatch.columns.values[0] = 'time(us)'
nobatch.columns.values[1] = 'gflops'
nobatch.columns.values[2] = 'GB/s'
nobatch['time(us)'] = nobatch['time(us)']-nobatch['time(us)'].min()
nobatch['time(us)'] = nobatch['time(us)']/1000000
nobatch.to_csv("nobatch.csv",index=False)