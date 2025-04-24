import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

left_bar_color = '#F7B236'
right_bar_color = '#FC5050'

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.usetex'] = False 

data = pd.read_csv("1.csv")

data = data.set_index('solver')

pangu_data = data.loc[['PanguLU420', 'PanguLU_sc']]

superlu_data = data.loc[['SuperLU910', 'SuperLU_sc']]

bar_width = 0.35
x = range(len(pangu_data.columns))

name_after = []
for name in pangu_data.columns:
    if name == 'Geometric Mean':
        name_after.append(name)
        continue
    if len(name) > 4 : 
        name_after.append(name[:4]+'...')
    else:
        name_after.append(name)

if 'Geometric Mean' in name_after:
    index = name_after.index('Geometric Mean')
    name_after[index] = 'Geometric\nMean'


fig, ax = plt.subplots(figsize=(16, 6))

ax.bar([i - bar_width/2 for i in x], superlu_data.loc['SuperLU910'], width=bar_width, label='SuperLU_DIST', color="#66C6BA", zorder=3)
bars_superlu_sc = ax.bar([i + bar_width/2 for i in x], superlu_data.loc['SuperLU_sc'], width=bar_width, label='SuperLU_DIST with Trojan Horse', color="#0092CA", zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(name_after, rotation=30, ha='right', fontsize=35)

for label in ax.get_xticklabels():
    if label.get_text() == 'Geometric\nMean':
        label.set_rotation(0)
        label.set_horizontalalignment('center') 
        label.set_y(-0.04)
    else:
        label.set_rotation(30)
        label.set_horizontalalignment('center') 
        label.set_y(0)

ax.set_ylabel('#Kernels (×10$^{6}$)', fontsize=40)
ax.set_xlabel('Matrices', fontsize=40, labelpad=0)

ax.grid(axis='y', linestyle='--', linewidth=2, color='gray', zorder=1)

ax.set_ylim(0, 7e6)
ax.set_yticks([0,2e6,4e6,6e6])
ax.set_yticklabels([0,2,4,6], fontsize=35)

for i, bar in enumerate(bars_superlu_sc):
    original_value = superlu_data.loc['SuperLU910'][i]
    new_value = superlu_data.loc['SuperLU_sc'][i]
    percentage_decrease = 100 - ((original_value - new_value) / original_value) * 100
    ax.text(bar.get_x() + bar.get_width()/1.8, bar.get_height(), f'{percentage_decrease:.1f}%', ha='center', va='bottom', fontsize=28)
    
    left_bar_height = superlu_data.loc['SuperLU910'][i]  
    right_bar_height = bar.get_height()  
    
legend = ax.legend(loc='upper left', ncol=1, fontsize=30)
legend.set_alpha(0.01) 
plt.tight_layout()
plt.savefig('SuperLU_kernelCount.pdf', bbox_inches='tight')
# plt.show()


fig, ax = plt.subplots(figsize=(16, 6))

ax.bar([i - bar_width/2 for i in x], pangu_data.loc['PanguLU420'], width=bar_width, label='PanguLU', color=left_bar_color, zorder=3)
bars_pangu_sc = ax.bar([i + bar_width/2 for i in x], pangu_data.loc['PanguLU_sc'], width=bar_width, label='PanguLU with Trojan Horse', color=right_bar_color, zorder=3)


ax.set_xticks(x)
ax.set_xticklabels(name_after, rotation=30, ha='right', fontsize=35)

for label in ax.get_xticklabels():
    if label.get_text() == 'Geometric\nMean':
        label.set_rotation(0)
        label.set_horizontalalignment('center') 
        label.set_y(-0.04)
    else:
        label.set_rotation(30)
        label.set_horizontalalignment('center') 
        label.set_y(0)

ax.set_ylabel('#Kernels (×10$^{4}$)', fontsize=40)

ax.set_xlabel('Matrices', fontsize=40, labelpad=0)


ax.grid(axis='y', linestyle='--', linewidth=2, color='gray', zorder=1)

ax.set_ylim(0, 10e4)
ax.set_yticks([0,3e4,6e4,9e4])
ax.set_yticklabels([0,3,6,9], fontsize=35)


for i, bar in enumerate(bars_pangu_sc):
    original_value = pangu_data.loc['PanguLU420'][i]
    new_value = pangu_data.loc['PanguLU_sc'][i]
    percentage_decrease = 100-((original_value - new_value) / original_value) * 100
    ax.text(bar.get_x() + bar.get_width()/1.8 , bar.get_height(), f'{percentage_decrease:.1f}%', ha='center', va='bottom', fontsize=28)
    
    left_bar_height = pangu_data.loc['PanguLU420'][i]  
    right_bar_height = bar.get_height()  
    
ax.legend(loc='upper left', ncol=1, fontsize=30)

plt.tight_layout()
plt.savefig('PanguLU_kernelCount.pdf', bbox_inches='tight')
# plt.show()
print("done")