import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def cut_name(matrix_names):
    name_after_cut = []
    for i in range(len(matrix_names)):
        if matrix_names[i] == 'Geometric Mean':
            name_after_cut.append('Geometric\nMean')
        elif(len(matrix_names[i])<=4):
            name_after_cut.append(matrix_names[i])
        else:
            name_after_cut.append(matrix_names[i][0:4]+'...')
    return name_after_cut

font_path = 'Arial.ttf'
font = FontProperties(fname=font_path)

title_fontsize = 40
label_fontsize = 40
tick_fontsize = 30
legend_fontsize = 24
fonty = FontProperties(fname=font_path, size=30)
font1 = FontProperties(fname=font_path, size=legend_fontsize)

data = pd.read_csv('superlu2-1.csv')
matrix_names = data['matrix']
pre_times = data['SuperLU910_numeric']
after_times = data['SuperLU_SC_numeric']
pre_components = data[['SuperLU910_kernel_after_add','SuperLU910_scatter']]

after_components = data[['SuperLU_SC_kernel']]

bar_width = 0.4
gap = 0
group_gap = 0.5
index = np.arange(0, len(matrix_names) * (1 + group_gap), 1 + group_gap)
colors = ['#e67133', '#9b9fee']

fig2 = plt.figure(figsize=(14, 6))

plt.bar(index - (bar_width + gap) / 2, pre_times, bar_width, color='#d3d4d8', label="Others (scheduling, etc.)", zorder=2)

label_origin = ['Kernel','Scatter']
bottom_pre = np.zeros(len(matrix_names))
for i, col in enumerate(pre_components.columns):
    bars = plt.bar(index - (bar_width + gap) / 2, pre_components[col], bar_width, 
                   bottom=bottom_pre, label=label_origin[i], color=colors[i], zorder=2)
    bottom_pre += 0

colors1 = ['#CF56A1']
plt.bar(index + (bar_width + gap) / 2, after_times, bar_width, color='#d3d4d8', zorder=2)
label_after = ['Trojan Horse batched Kernel']
for i, col in enumerate(after_components.columns):
    bars = plt.bar(index + (bar_width + gap) / 2, after_components[col], bar_width, 
                   bottom=bottom_pre, label=label_after[i], color=colors1[i], zorder=2)
    bottom_pre += 0

plt.xlabel('Matrices', fontsize=label_fontsize, fontproperties=font)
plt.ylabel('Numeric time (s)', fontsize=label_fontsize, fontproperties=font)
plt.xticks(index, cut_name(matrix_names), fontproperties=font, 
           fontsize=tick_fontsize, rotation=30, ha='right')
plt.yticks([0,2, 4, 6, 8, 10, 12, 14, 16])
for label in plt.gca().get_xticklabels():
    if label.get_text() == 'Geometric\nMean':
        label.set_rotation(0)
        label.set_horizontalalignment('center') 
        label.set_y(-0.04)

plt.grid(axis='y', linestyle='--', linewidth=2, color='gray', zorder=0)

for tick in plt.gca().yaxis.get_major_ticks():
    tick.label1.set_fontproperties(fonty)

plt.legend(prop=font1, handletextpad=0.5, labelspacing=0.05)

plt.tight_layout()
plt.savefig('SuperLU_compare.pdf', bbox_inches='tight')
# plt.show()


data = pd.read_csv('pangu2-1.csv')
matrix_names = data['matrix']
pre_times = data['Numeric factorization time-pre']
after_times = data['Numeric factorization time-after']
pre_components = data[['GETRF', 'TSTRF', 'GESSM', 'SSSSM']]

bar_width = 0.4
gap = 0
group_gap = 0.5
index = np.arange(0, len(matrix_names) * (1 + group_gap), 1 + group_gap)
colors = ['#ea5455', '#ffcd38', '#5be18a', '#417fe1']

fig2 = plt.figure(figsize=(14, 6))

plt.bar(index - (bar_width + gap) / 2, pre_times, bar_width, color='#d3d4d8', label="Others (scheduling, etc.) ", zorder=1)
label_pre = ['GETRF', 'TSTRF', 'GESSM', 'SSSSM']
bottom_pre = np.zeros(len(matrix_names))
for i, col in enumerate(pre_components.columns):
    bars = plt.bar(index - (bar_width + gap) / 2, pre_components[col], bar_width, 
                   bottom=bottom_pre, label=label_pre[i], color=colors[i], zorder=2)
    bottom_pre += 0

plt.bar(index + (bar_width + gap) / 2, after_times, bar_width, color='#cf56a1', 
        label='Trojan Horse batched Kernel', zorder=2)

plt.xlabel('Matrices', fontsize=label_fontsize, fontproperties=font)
plt.ylabel('Numeric time (s)', fontsize=label_fontsize, fontproperties=font)
plt.xticks(index, cut_name(matrix_names), fontproperties=font, 
           fontsize=tick_fontsize, rotation=30, ha='right')

for label in plt.gca().get_xticklabels():
    if label.get_text() == 'Geometric\nMean':
        label.set_rotation(0)
        label.set_horizontalalignment('center') 
        label.set_y(-0.04)

plt.grid(axis='y', linestyle='--', linewidth=2, color='gray', zorder=0)

for tick in plt.gca().yaxis.get_major_ticks():
    tick.label1.set_fontproperties(fonty)
plt.legend(prop=font1, handletextpad=0.5, labelspacing=0.05)

plt.tight_layout()
plt.savefig('PanguLU_compare.pdf', bbox_inches='tight')
# plt.show()