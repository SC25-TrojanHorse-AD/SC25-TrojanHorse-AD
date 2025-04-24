import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

data_4060 = pd.read_csv("data4060_with_gflops.csv")

data_4090 = pd.read_csv("data4090_with_gflops.csv")

valid_indices_pangu = (data_4060['PanguLU_4.2.0_original_4060'].notna() | data_4090['PanguLU_4.2.0_4090'].notna() |
                       data_4060['PanguLU_sc_4060'].notna() | data_4090['PanguLU_sc_4090'].notna())
valid_indices_superlu = (data_4060['superLU_original_4060'].notna() | data_4090['superLU_original_4090'].notna() |
                         data_4060['superLU_sc_4060'].notna() | data_4090['superLU_sc_4090'].notna())

pangu_original_4060 = data_4060[valid_indices_pangu]['PanguLU_4.2.0_original_4060']
pangu_sc_4060 = data_4060[valid_indices_pangu]['PanguLU_sc_4060']
superlu_original_4060 = data_4060[valid_indices_superlu]['superLU_original_4060']
superlu_sc_4060 = data_4060[valid_indices_superlu]['superLU_sc_4060']

pangu_original_4090 = data_4090[valid_indices_pangu]['PanguLU_4.2.0_4090']
pangu_sc_4090 = data_4090[valid_indices_pangu]['PanguLU_sc_4090']
superlu_original_4090 = data_4090[valid_indices_superlu]['superLU_original_4090']
superlu_sc_4090 = data_4090[valid_indices_superlu]['superLU_sc_4090']

bar_width = 0.35

fig, axes = plt.subplots(figsize=(21, 5))

index_4060_superlu = range(len(superlu_original_4060))
index_4090_superlu = [i + bar_width for i in index_4060_superlu]
bar_width1 = 0.35
axes.bar(index_4060_superlu, superlu_original_4060, bar_width1, label='SuperLU_DIST, RTX4060', color='#66C6BA', zorder=3)
axes.bar(index_4060_superlu, superlu_original_4090, bar_width, label='SuperLU_DIST, RTX4090', linewidth=1.5, color='#A4E5D9', zorder=2)
axes.bar(index_4090_superlu, superlu_sc_4060, bar_width1, label='SuperLU_DIST with Trojan Horse, RTX4060', color='#0092CA', zorder=3)
axes.bar(index_4090_superlu, superlu_sc_4090, bar_width, label='SuperLU_DIST with Trojan Horse, RTX4090', linewidth=1.5, color='#a3d7f9', zorder=2)

axes.set_xlabel('Matrices', fontsize=50, labelpad=0)
axes.set_ylabel('GFlops', fontsize=50)

name_list = data_4060[valid_indices_superlu]['matrix'].tolist()

index = [i - bar_width/3 for i in range(len(superlu_original_4060))]
index[-1] = 6.2
axes.set_xticks(index)
name_list[-1] = 'Geometric\nMean' 
for i in range(len(name_list)-1):
    if len(name_list[i]) > 4:
        name_list[i] = name_list[i][:4] + '...'
    else:
        name_list[i] = name_list[i]   
axes.set_xticklabels(name_list, rotation=30, fontsize=40)
for label in axes.get_xticklabels():
    if label.get_text() == 'Geometric\nMean':
        label.set_rotation(0)
        label.set_horizontalalignment('center') 
        label.set_y(-0.04)
    else:
        label.set_rotation(30)
        label.set_horizontalalignment('center') 
        label.set_y(0)

axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=30, handletextpad=0.5, columnspacing=0.5)

axes.tick_params(axis='y', labelsize=40)
axes.set_yticks([0, 20, 40, 60,80])

axes.grid(axis='y', linestyle='--', linewidth=2, color='gray', zorder=1)

plt.subplots_adjust(hspace=1.3)

plt.savefig('SuperLU_scale_up.pdf', format='pdf', bbox_inches='tight')
# plt.show()

fig, axes = plt.subplots(figsize=(21, 5))
bar_width1 = 0.35

index_4060 = range(len(pangu_original_4060))
index_4090 = [i + bar_width for i in index_4060]

axes.bar(index_4060, pangu_original_4060, bar_width1, label='PanguLU, RTX4060', color='#F7B236', zorder=3)
axes.bar(index_4060, pangu_original_4090, bar_width, label='PanguLU, RTX4090', linewidth=1.5, color='#FCE38A', zorder=2)
axes.bar(index_4090, pangu_sc_4060, bar_width1, label='PanguLU with Trojan Horse, RTX4060', color='#FC5050', zorder=3)
axes.bar(index_4090, pangu_sc_4090, bar_width, label='PanguLU with Trojan Horse, RTX4090', linewidth=1.5, color='#F6C7C7', zorder=2)

axes.set_xlabel('Matrices', fontsize=50, labelpad=0) 
axes.set_ylabel('GFlops', fontsize=50)

index = [i - bar_width/2 for i in range(len(pangu_original_4060))]
index[-1] = 6.2
axes.set_xticks(index)

name_list = data_4090[valid_indices_pangu]['matrix'].tolist()
name_list[-1] = 'Geometric\nMean'  
for i in range(len(name_list)-1):
    if len(name_list[i]) > 4:
        name_list[i] = name_list[i][:4] + '...'
    else:
        name_list[i] = name_list[i] 
axes.set_xticklabels(name_list, rotation=30, fontsize=40)
for label in axes.get_xticklabels():
    if label.get_text() == 'Geometric\nMean':
        label.set_rotation(0)
        label.set_horizontalalignment('center') 
        label.set_y(-0.04)

axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=2, fontsize=30 ,columnspacing=0.5, handletextpad=0.5)

axes.tick_params(axis='y', labelsize=40)
axes.set_yticks([0, 20, 40, 60,80])

axes.grid(axis='y', linestyle='--', linewidth=2, color='gray', zorder=0)

plt.subplots_adjust(hspace=1.3)

plt.savefig('Pangu_scale_up.pdf', format='pdf', bbox_inches='tight')
# plt.show()
