import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df1 = pd.read_csv('./GFlops/pangu_withGFlops.csv')
df2 = pd.read_csv('./GFlops/pangu_sc_withGFlops.csv')
df3 = pd.read_csv('./GFlops/superlu_withGFlops.csv')
df4 = pd.read_csv('./GFlops/superlu_sc_withGFlops.csv')
plt.rcParams['figure.dpi'] = 300

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
target_rows = ['GFflops_1', 'GFflops_2', 'GFflops_4', 'GFflops_8', 'GFflops_16', 'GFflops_32']

data1 = df2[df2['matrix'].isin(target_rows)]
data2 = df1[df1['matrix'].isin(target_rows)]
data3 = df3[df3['matrix'].isin(target_rows)]
data4 = df4[df4['matrix'].isin(target_rows)]

matrix_names = df1.columns.drop(['matrix']).tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes = axes.flatten()
matrix_names = ['cage12','Si87H76','RM07R','audikw_1','nlpkkt80','atmosmodd']

y_max_values = [1000, 2500, 1500, 1000, 1500, 1000]

for i, matrix_name in enumerate(matrix_names):
    y1 = data1[matrix_name]
    y2 = data2[matrix_name]
    y3 = data3[matrix_name]
    y4 = data4[matrix_name]

    x_positions = np.arange(6)

    axes[i].plot(x_positions, y3, label='SuperLU_DIST 9.1.0', marker='^', color='#66C6BA', linewidth=3, markersize=10)
    axes[i].plot(x_positions, y2, label='PanguLU 4.2.0', marker='s', color='#F7B236', linewidth=3, markersize=10)
    axes[i].plot(x_positions, y4, label='SuperLU_DIST 9.1.0 with Trojan Horse', marker='D', color='#0092CA', linewidth=3, markersize=10)
    axes[i].plot(x_positions, y1, label='PanguLU 4.2.0 with Trojan Horse', marker='o', color='#FC5050', linewidth=3, markersize=10)

    if i ==0:
        axes[i].set_xlabel(matrix_name, fontsize=40, labelpad=-283, x=0.29)
    if i ==1:
        axes[i].set_xlabel(matrix_name, fontsize=40, labelpad=-283, x=0.34)
    if i ==2:
        axes[i].set_xlabel(matrix_name, fontsize=40, labelpad=-283, x=0.31)
    if i == 3:
        axes[i].set_xlabel(matrix_name, fontsize=40, labelpad=-283, x=0.37)
    if i == 4:
        axes[i].set_xlabel(matrix_name, fontsize=40, labelpad=-283, x=0.34)
    if i == 5:
        axes[i].set_xlabel(matrix_name, fontsize=40, labelpad=-283, x=0.45)

    if i == 0 or i == 3:
        axes[i].set_ylabel('GFlops', fontsize=40)

    axes[i].set_xticks(x_positions)
    xticklabels = [1,2,4,8,16,32]
    xticklabels = [str(x) for x in xticklabels] 
    axes[i].set_xticklabels(xticklabels)

    axes[i].set_ylim(0, y_max_values[i])

    axes[i].grid(True, linestyle='--', color='gray', zorder=1)
    axes[i].tick_params(axis='x', labelsize=30)  
    axes[i].tick_params(axis='y', labelsize=27)  
    specific_yticks = [625,1250,1875, 2500]
    axes[1].set_yticks(specific_yticks)
    axes[1].set_yticklabels(specific_yticks, fontsize=27)  
    specific_yticks = [375,750,1125, 1500]
    axes[2].set_yticks(specific_yticks)
    axes[2].set_yticklabels(specific_yticks, fontsize=27) 
    specific_yticks = [375,750,1125, 1500]
    axes[4].set_yticks(specific_yticks)
    axes[4].set_yticklabels(specific_yticks, fontsize=27)  
    specific_yticks = [250,500,750, 1000]
    axes[5].set_yticks(specific_yticks)
    axes[5].set_yticklabels(specific_yticks, fontsize=27) 

handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=30, frameon=True, bbox_to_anchor=(0.52, 1.18), columnspacing=1.2, handletextpad=0.5) 

for line in legend.get_lines():
    line.set_linewidth(4) 
    line.set_markersize(16) 
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.2)

fig.text(0.545, -0.05, '#Processes (one GPU per process)', ha='center', fontsize=40)

plt.savefig("Scalability.pdf",bbox_inches='tight')
# plt.show()