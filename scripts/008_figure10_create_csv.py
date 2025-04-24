#!/usr/bin/python3
import sys
import os
import csv
ROOT_DIR = sys.argv[1]


def analyze_pangulu_kernelcnt_nohup(nohup_path):
    ret = []
    nohup = open(nohup_path,"r")
    nohup_lines = [x.strip() for x in nohup.readlines()]
    stat = 0
    matrix_name = ""
    kernel_count = -1
    for line in nohup_lines:
        if line.find("Reading") != -1:
            matrix_name = line.split(" ")[-1].split("/")[-1][:-4]
            stat = 1
        if line.find("kernel_count") != -1:
            if stat == 1:
                kernel_count = eval(line.split()[3])
                ret.append([matrix_name, kernel_count])
                stat = 0
    nohup.close()
    return ret

def analyze_superlu_kernelcnt_nohup(nohup_path):
    ret = []
    nohup = open(nohup_path,"r")
    nohup_lines = [x.strip() for x in nohup.readlines()]
    stat = 0
    matrix_name = ""
    kernel_count = -1
    for line in nohup_lines:
        if line.find("Input matrix file") != -1:
            matrix_name = line.split(" ")[-1].split("/")[-1][:-4]
            stat = 1
        if line.find("cpu_kernel_count") != -1:
            if stat == 1:
                kernel_count = eval(line.split()[1].split("=")[-1]) + eval(line.split()[2].split("=")[-1])
                ret.append([matrix_name, kernel_count])
                stat = 0
        if line.find(" kernel_count") != -1:
            if stat == 1:
                kernel_count = eval(line.split()[-1])
                ret.append([matrix_name, kernel_count])
                stat = 0
    nohup.close()
    return ret

def add_to_dict(list:list, dict:dict, solver:str):
    for item in list:
        matrix = item[0]
        kernel_count = item[1]
        if solver not in dict.keys():
            dict[solver] = {}
        dict[solver][matrix] = kernel_count


# print(analyze_pangulu_kernelcnt_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_420_kernel_cnt_and_time.txt")))
# print(analyze_pangulu_kernelcnt_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_sc25_kernel_cnt_and_time.txt")))

# 4090
pangulu_420_4090 = analyze_pangulu_kernelcnt_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_420_kernel_cnt_and_time.txt"))
pangulu_sc25_4090 = analyze_pangulu_kernelcnt_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_sc25_kernel_cnt_and_time.txt"))
superlu_910_4090 = analyze_superlu_kernelcnt_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_superlu_910_kernel_cnt_and_time.txt"))
superlu_sc25_4090 = analyze_superlu_kernelcnt_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_superlu_sc25_kernel_cnt_and_time.txt"))

data_4090 = {}
add_to_dict(pangulu_420_4090, data_4090, "PanguLU420")
add_to_dict(pangulu_sc25_4090, data_4090, "PanguLU_sc")
add_to_dict(superlu_910_4090, data_4090, "SuperLU910")
add_to_dict(superlu_sc25_4090, data_4090, "SuperLU_sc")

with open(os.path.join(ROOT_DIR, "Figures/Figure10/1.csv"), "w") as csvfile:
    fieldnames = ["solver","ex11","gas_sensor","shipsec1","para-8","inline_1","ldoor"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for solver in data_4090.keys():
        writer.writerow({**{"solver":solver}, **data_4090[solver]})