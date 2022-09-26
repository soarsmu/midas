from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import json
import os
import torch
from tqdm import tqdm
from torch import cuda
from torch import nn as nn
import matplotlib.pyplot as plt

directory = os.path.dirname(os.path.abspath(__file__))

dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset_new.csv'

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DATASET_FILE_FOLDER_PATH = os.path.join(directory, '../dataset_files')
DATASET_REMOVED_FILE_FOLDER_PATH = os.path.join(directory, '../dataset_removed_files')


def get_file_to_manual_map():
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo']]
    index_list = []
    for index, item in enumerate(df.values.tolist()):
        source_file_path = DATASET_FILE_FOLDER_PATH + '/' + str(index) + '.txt'
        removed_file_file_path = DATASET_REMOVED_FILE_FOLDER_PATH + '/' + str(index) + '.txt'
        if not os.path.isfile(source_file_path) and not os.path.isfile(removed_file_file_path):
            index_list.append(index)

    with open('missing_file_indices.txt', 'w') as writer:
        for index in index_list:
            writer.write(str(index) + '\n')


def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '/*',  '*', '*/', '#')) or line == '':
                continue
            code = code + line + '\n'

    return code


def manual_map():
    index_list = []
    with open('missing_file_indices.txt', 'r') as reader:
        lines = reader.read().split('\n')[:-1]
        for line in lines:
            index_list.append(int(line))

    df = pd.read_csv(dataset_name)
    df = df[['diff']]
    for index, item in enumerate(df.values.tolist()):
        if index not in index_list:
            continue
        diff = item[0]
        removed_code = get_code_version(diff, False)
        added_code = get_code_version(diff, True)

        removed_lines = removed_code.split('\n')[:-1]
        added_lines = added_code.split('\n')[:-1]
        print()


manual_map()