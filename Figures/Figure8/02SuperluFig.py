import os
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 6, figsize=(22, 2))

axes = axes.flatten() 

folders = [f for f in os.listdir('.') if os.path.isdir(f) and f.startswith('superlu_')]

print(folders)
folders = [ 'superlu_ex11','superlu_gas_sensor','superlu_shipsec1','superlu_para-8', 'superlu_inline_1','superlu_ldoor']

for i, folder in enumerate(folders):
    big_nobatch_path = os.path.join(folder, "nobatchMAX.xlsx")
    mid_nobatch_path = os.path.join(folder, "nobatchMEAN.xlsx")
    small_nobatch_path = os.path.join(folder, "nobatchMIN.xlsx")

    big_batch_path = os.path.join(folder, "batchMAX.xlsx")
    mid_batch_path = os.path.join(folder, "batchMEAN.xlsx")
    small_batch_path = os.path.join(folder, "batchMIN.xlsx")
    
    try:
        big_nobatch = pd.read_excel(big_nobatch_path)
        mid_nobatch = pd.read_excel(mid_nobatch_path)
        small_nobatch = pd.read_excel(small_nobatch_path)
        big_batch = pd.read_excel(big_batch_path)
        mid_batch = pd.read_excel(mid_batch_path)
        small_batch = pd.read_excel(small_batch_path)
        
        time_us_big_batch = big_batch["time(us)"]
        gflops_big_batch = big_batch["gflops"]
        time_us_mid_batch = mid_batch["time(us)"]
        gflops_mid_batch = mid_batch["gflops"]
        time_us_small_batch = small_batch["time(us)"]
        gflops_small_batch = small_batch["gflops"]
        time_us_big_nobatch = big_nobatch["time(us)"]
        gflops_big_nobatch = big_nobatch["gflops"]
        time_us_mid_nobatch = mid_nobatch["time(us)"]
        gflops_mid_nobatch = mid_nobatch["gflops"]
        time_us_small_nobatch = small_nobatch["time(us)"]
        gflops_small_nobatch = small_nobatch["gflops"]
        
        axes[i].plot(time_us_mid_nobatch, gflops_mid_nobatch, color='blue',zorder=2)
        axes[i].fill_between(time_us_big_nobatch, gflops_big_nobatch, gflops_small_nobatch, color='blue', alpha=0.3,label='SuperLU',zorder=1)
        axes[i].plot(time_us_mid_batch, gflops_mid_batch, color='red',zorder=2)
        axes[i].fill_between(time_us_big_batch, gflops_big_batch, gflops_small_batch, color='red', alpha=0.3,label='SuperLU with Trojan Horse',zorder=1)
        
        if folder[:8] == 'superlu_':
            folder = folder[8:]
        axes[i].set_title(folder, fontsize=20,pad=7)
        axes[i].set_xlabel('Time (s)', fontsize=20,labelpad=1)
        if i % 8 == 0:
            axes[i].set_ylabel('GFlops', fontsize=20)
        
        time_max = max([max(time_us_big_batch), max(time_us_mid_batch), max(time_us_small_batch),
                        max(time_us_big_nobatch), max(time_us_mid_nobatch), max(time_us_small_nobatch)])
        
        axes[i].set_xlim(left=0, right=time_max)
        gflops_max = max([max(gflops_big_batch), max(gflops_small_batch), max(gflops_mid_batch),
                          max(gflops_big_nobatch), max(gflops_small_nobatch), max(gflops_mid_nobatch)])
        axes[i].set_ylim(bottom=0, top=gflops_max)
        
        time_max = float(time_max)+float(time_max)/20
        gflops_max = int(gflops_max)+1
        axes[i].set_ylim(bottom=0, top=gflops_max)
        axes[i].set_xlim(left=0, right=time_max)
        meta = float(f'{time_max / 4:.1f}')
        meta2 = float(f'{2*meta:.1f}')
        meta3 = float(f'{3*meta:.1f}')

        axes[0].set_xticks(ticks=[0, 0.1, 0.2,0.3])
        axes[0].set_xticklabels([0, 0.1, 0.2,0.3], fontsize=20)
        
        axes[1].set_xticks(ticks=[0,  1,2])
        axes[1].set_xticklabels([0,  1,2], fontsize=20)

        axes[2].set_xticks(ticks=[0, 1,2,3,4])
        axes[2].set_xticklabels([0, 1,2,3,4], fontsize=20)

        axes[3].set_xticks(ticks=[0, 5,10,15])
        axes[3].set_xticklabels([0,5,10,15], fontsize=20)

        axes[4].set_xticks(ticks=[0, 5,10,15])
        axes[4].set_xticklabels([0,5,10,15], fontsize=20)

        axes[5].set_xticks(ticks=[0,2.5,5,7.5,10])
        axes[5].set_xticklabels([0,2.5,5,7.5,10], fontsize=20)

        axes[i].set_yticks(ticks=[0, int(gflops_max/3), int(gflops_max/3)*2,int(gflops_max/3)*3])
        axes[i].set_yticklabels([0, int(gflops_max/3), int(gflops_max/3)*2, int(gflops_max/3)*3], fontsize=20)
    except FileNotFoundError:
        print(f" {folder} not found.")

legend = axes[3].legend(loc='upper center', fontsize=22,ncol=4,bbox_to_anchor=(-0.4, 1.7),columnspacing=1)

plt.subplots_adjust(wspace=0.4, hspace=0.8)  
plt.savefig("batch_compare_superlu.pdf", dpi=300, bbox_inches='tight') 
# plt.show()