#!/usr/bin/python3
import sys
import os
import csv
ROOT_DIR = sys.argv[1]


def analyze_pangulu_kernel_breakdown_nohup(nohup_path):
    ret = []
    nohup = open(nohup_path,"r")
    nohup_lines = [x.strip() for x in nohup.readlines()]
    stat = 0
    ret_item = ["",0]
    for line in nohup_lines:
        if line.find("Reading") != -1:
            ret_item[0] = line.split(" ")[-1].split("/")[-1][:-4]
            stat = 1
        if line.find("Numeric factorization time") != -1:
            if stat == 1:
                ret_item[1] = eval(line.split()[-2])
                stat = 2
        if line.find("time_getrf=") != -1:
            if stat == 2:
                ret_item.append(eval(line.split()[4].split("=")[-1]))
                ret_item.append(eval(line.split()[5].split("=")[-1]))
                ret_item.append(eval(line.split()[6].split("=")[-1]))
                ret_item.append(eval(line.split()[7].split("=")[-1]))
                ret.append(ret_item)
                ret_item = ["",0]
                stat = 0
        if line.find("#0	0.000000	0.000000	0.000000") != -1:
            if stat == 2:
                ret_item.append(eval(line.split()[-1]))
                ret.append(ret_item)
                ret_item = ["",0]
                stat = 0
    nohup.close()
    return ret

def analyze_superlu_kernel_breakdown_nohup(nohup_path):
    ret = []
    nohup = open(nohup_path,"r")
    nohup_lines = [x.strip() for x in nohup.readlines()]
    stat = 0
    ret_item = ["",0]
    kernel_count = -1
    for line in nohup_lines:
        if line.find("Input matrix file") != -1:
            ret_item[0] = line.split(" ")[-1].split("/")[-1][:-4]
            stat = 1
        if line.find("cpu_kernel_time") != -1:
            if stat == 1:
                ret_item.append(eval(line.split()[3].split("=")[-1]) + eval(line.split()[4].split("=")[-1]))
                ret_item.append(eval(line.split()[5].split("=")[-1]))
                stat = 2
        if line.find(" kernel_time") != -1:
            if stat == 1:
                ret_item.append(eval(line.split()[3]))
                stat = 2
        if line.find("FACTOR time") != -1:
            if stat == 2:
                ret_item[1] = eval(line.split()[-1])
                ret.append(ret_item)
                ret_item = ["",0]
                stat = 0
    nohup.close()
    return ret

def add_to_dict_pangulu(list:list, dict:dict):
    for item in list:
        matrix = item[0]
        if matrix not in dict.keys():
            dict[matrix] = {}
        if len(item) == 3:
            dict[matrix]["Numeric factorization time-after"] = item[1]
        else:
            dict[matrix]["Numeric factorization time-pre"] = item[1]
            dict[matrix]["GETRF"] = item[2]
            dict[matrix]["TSTRF"] = item[3]
            dict[matrix]["GESSM"] = item[4]
            dict[matrix]["SSSSM"] = item[5]

def add_to_dict_superlu(list:list, dict:dict):
    for item in list:
        matrix = item[0]
        if matrix not in dict.keys():
            dict[matrix] = {}
        if len(item) == 3:
            dict[matrix]["SuperLU_SC_numeric"] = item[1]
            dict[matrix]["SuperLU_SC_kernel"] = item[2]
        else:
            dict[matrix]["SuperLU910_numeric"] = item[1]
            dict[matrix]["SuperLU910_kernel"] = item[2]
            dict[matrix]["SuperLU910_scatter"] = item[3]


# 4090
pangulu_420_4090 = analyze_pangulu_kernel_breakdown_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_420_kernel_cnt_and_time.txt"))
pangulu_sc25_4090 = analyze_pangulu_kernel_breakdown_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_sc25_kernel_cnt_and_time.txt"))
superlu_910_4090 = analyze_superlu_kernel_breakdown_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_superlu_910_kernel_cnt_and_time.txt"))
superlu_sc25_4090 = analyze_superlu_kernel_breakdown_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_superlu_sc25_kernel_cnt_and_time.txt"))

data_pangulu = {}
add_to_dict_pangulu(pangulu_420_4090, data_pangulu)
add_to_dict_pangulu(pangulu_sc25_4090, data_pangulu)
data_superlu = {}
add_to_dict_superlu(superlu_910_4090, data_superlu)
add_to_dict_superlu(superlu_sc25_4090, data_superlu)

with open(os.path.join(ROOT_DIR, "Figures/Figure11/pangu2.csv"), "w") as csvfile:
    fieldnames = ["matrix","Numeric factorization time-pre","GETRF","TSTRF","GESSM","SSSSM","Numeric factorization time-after"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for matrix in data_pangulu.keys():
        writer.writerow({**{"matrix":matrix}, **data_pangulu[matrix]})

with open(os.path.join(ROOT_DIR, "Figures/Figure11/superlu2.csv"), "w") as csvfile:
    fieldnames = ["matrix","SuperLU910_numeric","SuperLU910_kernel","SuperLU910_scatter","SuperLU_SC_numeric","SuperLU_SC_kernel"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for matrix in data_superlu.keys():
        writer.writerow({**{"matrix":matrix}, **data_superlu[matrix]})