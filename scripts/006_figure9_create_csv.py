#!/usr/bin/python3
import sys
import os
import csv
ROOT_DIR = sys.argv[1]

def analyze_pangulu_benchmark_nohup(nohup_path):
    ret = []
    nohup = open(nohup_path,"r")
    nohup_lines = [x.strip() for x in nohup.readlines()]
    stat = 0
    matrix_name = ""
    numeric_time = -1
    for line in nohup_lines:
        if line.find("Reading") != -1:
            matrix_name = line.split(" ")[-1].split("/")[-1][:-4]
            stat = 1
        if line.find("Numeric factorization time") != -1:
            if stat == 1:
                numeric_time = eval(line.split()[-2])
                ret.append([matrix_name, numeric_time])
                stat = 0
    nohup.close()
    return ret

def analyze_superlu_benchmark_nohup(nohup_path):
    ret = []
    nohup = open(nohup_path,"r")
    nohup_lines = [x.strip() for x in nohup.readlines()]
    stat = 0
    matrix_name = ""
    numeric_time = -1
    for line in nohup_lines:
        if line.find("Input matrix file") != -1:
            matrix_name = line.split(" ")[-1].split("/")[-1][:-4]
            stat = 1
        if line.find("FACTOR") != -1:
            if stat == 1:
                numeric_time = eval(line.split()[-1])
                ret.append([matrix_name, numeric_time])
                stat = 0
    nohup.close()
    return ret

def add_to_dict(list:list, dict:dict, fieldname:str):
    for item in list:
        matrix = item[0]
        numeric_time = item[1]
        if matrix not in dict.keys():
            dict[matrix] = {}
        dict[matrix][fieldname] = numeric_time

# 4060
pangulu_420_4060 = analyze_pangulu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu_4060/nohup_pangulu_420_benchmark.txt"))
pangulu_sc25_4060 = analyze_pangulu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu_4060/nohup_pangulu_sc25_benchmark.txt"))
superlu_910_4060 = analyze_superlu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu_4060/nohup_superlu_910_benchmark.txt"))
superlu_sc25_4060 = analyze_superlu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu_4060/nohup_superlu_sc25_benchmark.txt"))

data_4060 = {}
add_to_dict(pangulu_420_4060, data_4060, "PanguLU_4.2.0_original_4060")
add_to_dict(pangulu_sc25_4060, data_4060, "PanguLU_sc_4060")
add_to_dict(superlu_910_4060, data_4060, "superLU_original_4060")
add_to_dict(superlu_sc25_4060, data_4060, "superLU_sc_4060")

with open(os.path.join(ROOT_DIR, "Figures/Figure9/4060.csv"), "w") as csvfile:
    fieldnames = ["matrix","PanguLU_4.2.0_original_4060","PanguLU_sc_4060","superLU_original_4060","superLU_sc_4060"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for matrix in data_4060.keys():
        writer.writerow({**{"matrix":matrix}, **data_4060[matrix]})

# 4090
pangulu_420_4090 = analyze_pangulu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_420_benchmark.txt"))
pangulu_sc25_4090 = analyze_pangulu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_pangulu_sc25_benchmark.txt"))
superlu_910_4090 = analyze_superlu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_superlu_910_benchmark.txt"))
superlu_sc25_4090 = analyze_superlu_benchmark_nohup(os.path.join(ROOT_DIR, "results_singlegpu/nohup_superlu_sc25_benchmark.txt"))

data_4090 = {}
add_to_dict(pangulu_420_4090, data_4090, "PanguLU_4.2.0_4090")
add_to_dict(pangulu_sc25_4090, data_4090, "PanguLU_sc_4090")
add_to_dict(superlu_910_4090, data_4090, "superLU_original_4090")
add_to_dict(superlu_sc25_4090, data_4090, "superLU_sc_4090")

with open(os.path.join(ROOT_DIR, "Figures/Figure9/4090.csv"), "w") as csvfile:
    fieldnames = ["matrix","PanguLU_4.2.0_4090","PanguLU_sc_4090","superLU_original_4090","superLU_sc_4090"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for matrix in data_4090.keys():
        writer.writerow({**{"matrix":matrix}, **data_4090[matrix]})