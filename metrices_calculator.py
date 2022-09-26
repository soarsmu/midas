from resource import getrusage
from termios import FF1
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
import random
from transformers import AdamW
from transformers import get_scheduler
import csv
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
# from tse_experiments import get_url_to_hunk_count

is_test = False
test_size = 10000

directory = os.path.dirname(os.path.abspath(__file__))

dataset_name = 'ase_dataset_sept_19_2021.csv'
sec_dataset_name = 'ase_surity_sub_dataset.csv'

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')


DATA_LOADER_PARAMS = {'batch_size': 128, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 512
HIDDEN_DIM = 768

NEED_EXPANDED_VERSION = True

FIRST_PHASE_NUMBER_OF_EPOCHS = 6
NUMBER_OF_EPOCHS = 15

HIDDEN_DIM_DROPOUT_PROB = 0.3
NUMBER_OF_LABELS = 2
NUM_HEADS = 8

cos = nn.CosineSimilarity(dim=1)

MAX_VULN_LOC = 308

HUNK_COUNT_THRESHOLD = 5

def get_url_to_loc(item_list, url_to_hunk_count=None):
    url_to_loc = {}

    for item in item_list:
        url = item[0] + '/commit/' + item[1]
        if url_to_hunk_count is None or (url_to_hunk_count is not None and url_to_hunk_count[url] >= 7):
            loc = item[2]
            if url not in url_to_loc:
                url_to_loc[url] = 0
            url_to_loc[url] += loc

    return url_to_loc


def get_effort_hunk_or_file(df, lang, threshold, predicted, url_to_hunk_count):
    print("Calculating effort...")
    df = df[df.PL == lang]
    df_pos = df[df.label == 1]
    df_pos = df_pos[['repo', 'commit_id', 'LOC_MOD']]

    df_neg = df[df.label == 0]
    df_neg = df_neg[['repo', 'commit_id', 'LOC_MOD']]

    pos_items = df_pos.values.tolist()
    neg_items = df_neg.values.tolist()
    
    total_hunk = 0

    url_set = set()
    
    total_vulnerabilities = 0 

    for item in pos_items:
        url = item[0] + '/commit/' + item[1]
        if url not in url_set:
            total_vulnerabilities += 1
        url_set.add(item[0] + '/commit/' + item[1])
    for item in neg_items:
        url_set.add(item[0] + '/commit/' + item[1])

    for url in url_set:
        total_hunk += url_to_hunk_count[url]

    count = 0
    
    # for index, loc in enumerate(pos_loc_list):
    #     if (count + loc) / total_loc <= threshold:
    #         count += loc
    #     else:
    #         total_vulnerabilities = index - 1
    #         break

    #     if index == len(pos_loc_list) - 1 and (count + loc) / total_loc <= threshold:
    #         total_vulnerabilities = len(pos_loc_list)

    total_inspected = 0
    detected_vulnerabilities = 0
    predicted_indices = []
    non_vul_indices = []
    commit_count = 0
    ifa = len(predicted)
    found_vuln = False
    for index, item in enumerate(predicted):
        commit_index = item[0]
        hunk = item[2]
        label = item[3]
        rate = (total_inspected + hunk) / total_hunk
        if rate <= threshold:
            commit_count += 1
            total_inspected += hunk
            if label == 1:
                if not found_vuln:
                    ifa = commit_count - 1
                    found_vuln = True
                detected_vulnerabilities += 1
                predicted_indices.append(commit_index)
            else:
                non_vul_indices.append(commit_index)
        else:
            break
    recall = detected_vulnerabilities / total_vulnerabilities
    precision = detected_vulnerabilities / commit_count
    f1 = 2 * (precision * recall) / (precision + recall)
    pci = commit_count / len(predicted)

    print("Commit count: {}".format(commit_count))
    print("Vulnerability found: {}".format(detected_vulnerabilities))
    print("Total vulnerabilities: {}".format(total_vulnerabilities))
    # print("Precision: {}".format(precision))
    # print("Recall: {}".format(recall))
    # print("F1: {}".format(f1))
    # print("PCI: {}". format(pci))
    # print("IFA: {}".format(ifa))

    return detected_vulnerabilities / total_vulnerabilities, predicted_indices, non_vul_indices



def get_effort(df, lang, threshold, predicted, url_to_hunk_count=None):
    # print("Calculating effort...")
    df = df[df.PL == lang]
    df_pos = df[df.label == 1]
    df_pos = df_pos[['repo', 'commit_id', 'LOC_MOD']]

    df_neg = df[df.label == 0]
    df_neg = df_neg[['repo', 'commit_id', 'LOC_MOD']]

    pos_items = df_pos.values.tolist()
    neg_items = df_neg.values.tolist()

    pos_url_to_loc = get_url_to_loc(pos_items, url_to_hunk_count)
    neg_url_to_loc = get_url_to_loc(neg_items, url_to_hunk_count)

    total_commits = len(pos_url_to_loc) + len(neg_url_to_loc)

    pos_loc_list = list(pos_url_to_loc.values())
    neg_loc_list = list(neg_url_to_loc.values())

    pos_loc_list.sort()
    neg_loc_list.sort()

    total_loc = 0
    for loc in pos_loc_list:
        total_loc += loc
    for loc in neg_loc_list:
        total_loc += loc

    count = 0
    total_vulnerabilities = 0
    for index, loc in enumerate(pos_loc_list):
        if (count + loc) / total_loc <= threshold:
            count += loc
        else:
            total_vulnerabilities = index - 1
            break

        if index == len(pos_loc_list) - 1 and (count + loc) / total_loc <= threshold:
            total_vulnerabilities = len(pos_loc_list)

    total_inspected = 0
    detected_vulnerabilities = 0
    predicted_indices = []
    non_vul_indices = []
    commit_count = 0
    ifa = len(predicted)
    found_vuln = False
    for index, item in enumerate(predicted):
        commit_index = item[0]
        loc = item[2]
        label = item[3]
        rate = (total_inspected + loc) / total_loc
        if rate <= threshold:
            commit_count += 1
            total_inspected += loc
            if label == 1:
                if not found_vuln:
                    ifa = commit_count - 1
                    found_vuln = True
                detected_vulnerabilities += 1
                predicted_indices.append(commit_index)
            else:
                non_vul_indices.append(commit_index)
        else:
            break
    recall = detected_vulnerabilities / total_vulnerabilities
    precision = detected_vulnerabilities / commit_count
    f1 = 2 * (precision * recall) / (precision + recall)
    pci = commit_count / len(predicted)

    # print("Total commit: {}".format(total_commits))
    # print("Commit count: {}".format(commit_count))
    # print("Percentage commit: {}".format(commit_count/total_commits))
    # print("Vulnerability found: {}".format(detected_vulnerabilities))
    # print("Total vulnerabilities: {}".format(total_vulnerabilities))
    # print("Precision: {}".format(precision))
    # print("Recall: {}".format(recall))
    # print("F1: {}".format(f1))
    # print("PCI: {}". format(pci))
    # print("IFA: {}".format(ifa))

    return detected_vulnerabilities / total_vulnerabilities, predicted_indices, non_vul_indices


def get_recall_effort(threshold, predicted, total_vul):
    print("Calculating effort...")

    detected_vulnerabilities = 0
    predicted_indices = []
    non_vul_indices = []
    commit_count = 0
    total_commit = len(predicted)
    need_print = False
    for index, item in enumerate(predicted):
        commit_index = item[0]
        label = item[3]
        rate = commit_count/total_commit
        if rate < threshold:
            commit_count += 1
            if need_print and rate > 0.01:      # just for debugging
                need_print = False
                print("commit count {}".format(commit_count))
            if label == 1:
                detected_vulnerabilities += 1
                predicted_indices.append(commit_index)
            else:
                non_vul_indices.append(commit_index)
        else:
            break

    return detected_vulnerabilities / total_vul, predicted_indices, non_vul_indices



def calculate_effort(predicted_path, lang, url_to_hunk_count=None, url_set=None):
    # print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']

    url_to_label, url_to_loc_mod, _, _ = get_security_data(dataset_name)


    # row[0] : index => need to replace with url
    # row[1] : pred_prob

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            url = row[0]

            if url_set is not None and url not in url_set:
                continue

            items.append((row[0], float(row[1])))

    predicted = []
    url_to_pred = {}
    for item in items:
        predicted.append((item[0], item[1], url_to_loc_mod[item[0]], url_to_label[item[0]]))
        url_to_pred[item[0]] = item[1]
    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    # effort_5, _, _ = get_effort(df, lang, 0.01, predicted)

    effort_1, predicted_indices, non_vul_indices = get_effort(df, lang, 0.05, predicted, url_to_hunk_count)
    # with open(predicted_path + '_predicted_indices_cost_effort_5.txt', 'w') as file:
    #     writer = csv.writer(file)
    #     for url in predicted_indices:
    #         writer.writerow([url])

    effort_2, predicted_indices, _ = get_effort(df, lang, 0.1, predicted, url_to_hunk_count)
    # with open(predicted_path + '_predicted_indices_cost_effort_10.txt', 'w') as file:
    #     writer = csv.writer(file)
    #     for url in predicted_indices:
    #         writer.writerow([url])
    
    effort_3, predicted_indices, _ = get_effort(df, lang, 0.15, predicted, url_to_hunk_count)
    # with open(predicted_path + '_predicted_indices_cost_effort_15.txt', 'w') as file:
    #     writer = csv.writer(file)
    #     for url in predicted_indices:
    #         writer.writerow([url])

    effort_4, predicted_indices, _ = get_effort(df, lang, 0.20, predicted, url_to_hunk_count)
    # with open(predicted_path + '_predicted_indices_cost_effort_20.txt', 'w') as file:
    #     writer = csv.writer(file)
    #     for url in predicted_indices:
    #         writer.writerow([url])

    # print("Effort 1%: {}".format(effort_5))
    print("Effort 5%: {}".format(effort_1))
    print("Effort 10%: {}".format(effort_2))
    print("Effort 15%: {}".format(effort_3))
    print("Effort 20%: {}".format(effort_4))

    return predicted_indices


def calculate_effort_hunk(predicted_path, lang):
    print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']

    url_to_label, url_to_loc_mod = get_data()
    url_to_hunk_count = get_url_to_hunk_count()
    # row[0] : index => need to replace with url
    # row[1] : pred_prob

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append((row[0], float(row[1])))

    predicted = []
    url_to_pred = {}
    for item in items:
        predicted.append((item[0], item[1], url_to_hunk_count[item[0]], url_to_label[item[0]]))
        url_to_pred[item[0]] = item[1]
    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    effort_1, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.05, predicted, url_to_hunk_count)
    effort_2, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.1, predicted, url_to_hunk_count)
    effort_3, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.15, predicted, url_to_hunk_count)
    effort_4, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.20, predicted, url_to_hunk_count)

    print("CostEffort@5%: {}".format(effort_1))
    print("CostEffort@10%: {}".format(effort_2))
    print("CostEffort@15%: {}".format(effort_3))
    print("CostEffort@20%: {}".format(effort_4))

    return predicted_indices


def calculate_effort_file(predicted_path, lang):
    print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']

    url_to_label, url_to_loc_mod, url_to_file_count = get_data(need_file_count=True)
    # row[0] : index => need to replace with url
    # row[1] : pred_prob

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append((row[0], float(row[1])))

    predicted = []
    url_to_pred = {}
    for item in items:
        predicted.append((item[0], item[1], url_to_file_count[item[0]], url_to_label[item[0]]))
        url_to_pred[item[0]] = item[1]
    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    effort_1, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.05, predicted, url_to_file_count)
    effort_2, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.1, predicted, url_to_file_count)
    effort_3, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.15, predicted, url_to_file_count)
    effort_4, predicted_indices, _ = get_effort_hunk_or_file(df, lang, 0.20, predicted, url_to_file_count)

    print("Effort 5%: {}".format(effort_1))
    print("Effort 10%: {}".format(effort_2))
    print("Effort 15%: {}".format(effort_3))
    print("Effort 20%: {}".format(effort_4))

    return predicted_indices

def calculate_recall_effort(predicted_path, lang):
    url_to_label, url_to_loc_mod = get_data()

    total_vul = 300
    if lang == 'python':
        total_vul = 195

    print("Total vulnerabilities: {}".format(total_vul))

    # row[0] : index => need to replace with url
    # row[1] : pred_prob

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            items.append((row[0], float(row[1])))

    predicted = []
    url_to_pred = {}
    for item in items:
        predicted.append((item[0], item[1], url_to_loc_mod[item[0]], url_to_label[item[0]]))
        url_to_pred[item[0]] = item[1]
    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    effort_1, predicted_indices, non_vul_indices = get_recall_effort(0.05, predicted, total_vul)
    effort_2, predicted_indices, non_vul_indices = get_recall_effort(0.10, predicted, total_vul)
    effort_3, predicted_indices, non_vul_indices = get_recall_effort(0.15, predicted, total_vul)
    effort_4, predicted_indices, non_vul_indices = get_recall_effort(0.20, predicted, total_vul)
    print("Effort 5%: {}".format(effort_1))
    print("Effort 10%: {}".format(effort_2))
    print("Effort 15%: {}".format(effort_3))
    print("Effort 20%: {}".format(effort_4))

    return predicted_indices

def get_normalized_effort(df, lang, threshold, predicted, url_to_hunk_count=None):
    # print("Calculating effort...")
    df = df[df.PL == lang]
    df_pos = df[df.label == 1]
    df_pos = df_pos[['repo', 'commit_id', 'LOC_MOD']]

    df_neg = df[df.label == 0]
    df_neg = df_neg[['repo', 'commit_id', 'LOC_MOD']]

    pos_items = df_pos.values.tolist()
    neg_items = df_neg.values.tolist()

    pos_url_to_loc = get_url_to_loc(pos_items, url_to_hunk_count)
    neg_url_to_loc = get_url_to_loc(neg_items, url_to_hunk_count)

    pos_loc_list = list(pos_url_to_loc.values())
    neg_loc_list = list(neg_url_to_loc.values())

    pos_loc_list.sort()
    neg_loc_list.sort()

    total_loc = 0
    for loc in pos_loc_list:
        total_loc += loc
    for loc in neg_loc_list:
        total_loc += loc

    # Calculate AUC of optimal model

    count = 0
    x_optimal = []
    y_optimal = []
    x_optimal.append(0)
    y_optimal.append(0)

    for index, loc in enumerate(pos_loc_list):
        if (count + loc) / total_loc <= threshold:
            count += loc
            x_optimal.append(count / total_loc)
            y_optimal.append((index + 1) / len(pos_loc_list))
        else:
            break

    if (count + neg_loc_list[0]) / total_loc <= threshold:
        for loc in neg_loc_list:
            if (count + loc) / total_loc <= threshold:
                count += loc
                x_optimal.append(count / total_loc)
                y_optimal.append(1)

    auc_optimal = metrics.auc(x=x_optimal, y=y_optimal)

    # Calculate AUC of worst model

    count = 0
    x_worst = []
    y_worst = []
    x_worst.append(0)
    y_worst.append(0)
    neg_loc_list.reverse()
    pos_loc_list.reverse()

    for loc in neg_loc_list:
        if (count + loc) / total_loc <= threshold:
            count += 0
            x_worst.append(count / total_loc)
            y_worst.append(0)
        else:
            break

    if (count + pos_loc_list[0]) / total_loc <= threshold:
        for index, loc in enumerate(pos_loc_list):
            if (count + loc) / total_loc <= threshold:
                count += loc
                x_optimal.append(count / total_loc)
                y_optimal.append((index + 1) / len(pos_loc_list))

    auc_worst = metrics.auc(x=x_worst, y=y_worst)

    # Calculate AUC of model

    total_inspected = 0
    detected_vulnerabilities = 0
    x_model = []
    y_model = []
    x_model.append(0)
    y_model.append(0)
    commit_count = 0
    need_print = True
    for index, item in enumerate(predicted):
        loc = item[2]
        label = item[3]
        commit_count += 1
        if need_print and (total_inspected + loc) / total_loc > 0.01:
            need_print = False
            # print("Commit count: {}".format(commit_count))
        if (total_inspected + loc) / total_loc <= threshold:
            total_inspected += loc
            x_model.append(total_inspected / total_loc)
            if label == 1:
                detected_vulnerabilities += 1
            y_model.append(detected_vulnerabilities / len(pos_loc_list))
        else:
            break

    auc_model = metrics.auc(x=x_model, y=y_model)

    # plt.plot(x_model, y_model)
    # plt.show()

    # with open("model_popt.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     for i, x in enumerate(x_model):
    #         writer.writerow([x, y_model[i]])

    result = (auc_model - auc_worst) / (auc_optimal - auc_worst)

    return result


def get_security_data(dataset_name, need_file_count=False):
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD']]
    items = df.to_numpy().tolist()
    java_url_set = set()
    python_url_set = set()

    url_to_diff = {}
    url_to_label = {}
    url_to_pl = {}
    url_to_loc_mod = {}
    url_to_file_count = {}
    
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        label = item[4]
        pl = item[5]
        loc_mod = item[6]

        if url not in url_to_diff:
            url_to_diff[url] = ''
            url_to_loc_mod[url] = 0
            url_to_file_count[url] = 0

        url_to_label[url] = label
        url_to_pl[url] = pl
        url_to_loc_mod[url] += loc_mod
        url_to_file_count[url] += 1

        if pl == 'java':
            java_url_set.add(url)
        else:
            python_url_set.add(url)

    if need_file_count:
        return url_to_label, url_to_loc_mod, url_to_file_count, java_url_set, python_url_set
    else:
        return url_to_label, url_to_loc_mod, java_url_set, python_url_set


def get_data(need_file_count=False):
    # print("Reading dataset...")
    if os.path.isfile("url_to_loc.txt") and os.path.isfile("url_to_file_count.txt"):
        url_to_label = {}
        url_to_loc_mod = {}
        url_to_file_count = {}
        df = pd.read_csv("url_to_loc.txt", header=None)
        for item in df.values.tolist():
            url_to_label[item[0]] = item[1]
            url_to_loc_mod[item[0]] = item[2]

        df = pd.read_csv("url_to_file_count.txt", header=None)
        for item in df.values.tolist():
            url_to_label[item[0]] = item[1]
            url_to_file_count[item[0]] = item[2]

        if need_file_count:
            return url_to_label, url_to_loc_mod, url_to_file_count
        else:
            return url_to_label, url_to_loc_mod


    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_label = {}
    url_to_pl = {}
    url_to_loc_mod = {}
    url_to_file_count = {}
    
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        label = item[4]
        pl = item[5]
        loc_mod = item[6]

        if url not in url_to_diff:
            url_to_diff[url] = ''
            url_to_loc_mod[url] = 0
            url_to_file_count[url] = 0

        url_to_label[url] = label
        url_to_pl[url] = pl
        url_to_loc_mod[url] += loc_mod
        url_to_file_count[url] += 1

    with open("url_to_loc.txt", 'w') as file:
        writer = csv.writer(file)
        for url, label in url_to_label.items():
            writer.writerow([url, label, url_to_loc_mod[url]])

    with open("url_to_file_count.txt", 'w') as file:
        writer = csv.writer(file)
        for url, label in url_to_label.items():
            writer.writerow([url, label, url_to_file_count[url]])

    if need_file_count:
        return url_to_label, url_to_loc_mod, url_to_file_count
    else:
        return url_to_label, url_to_loc_mod


def calculate_normalized_effort(predicted_path, lang, url_to_hunk_count=None, url_set=None):
    # print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']

    url_to_label, url_to_loc_mod, _, _ = get_security_data(dataset_name)

    items = []
    with open(predicted_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:

            url = row[0]

            if url_set is not None and url not in url_set:
                continue

            items.append((row[0], float(row[1])))

    predicted = []
    for item in items:
        predicted.append((item[0], item[1], url_to_loc_mod[item[0]], url_to_label[item[0]]))

    predicted = sorted(predicted, key=lambda x: (-x[1], x[2]))

    print("Popt@5%: {}".format(get_normalized_effort(df, lang, 0.05, predicted, url_to_hunk_count)))
    print("Popt@10%: {}".format(get_normalized_effort(df, lang, 0.10, predicted, url_to_hunk_count)))
    print("Popt@15%: {}".format(get_normalized_effort(df, lang, 0.15, predicted, url_to_hunk_count)))
    print("Popt@20%: {}".format(get_normalized_effort(df, lang, 0.2, predicted, url_to_hunk_count)))


def write_predicted_indices_to_file(result_file_path, predicted_indices_file_path):
    indices = calculate_effort(result_file_path, 'python')
    with open(predicted_indices_file_path, 'w') as writer:
        for index in indices:
            writer.write(str(index) + '\n')


# pure_classifier_java_file_path = 'huawei_pure_classifier_prob_java.txt'
# pure_classifier_python_file_path = 'huawei_pure_classifier_prob_python.txt'
# comparison_classifier_java_file_path = 'huawei_comparison_classifier_prob_java.txt'
# comparison_classifier_python_filepath = 'huawei_comparison_classifier_prob_python.txt'
# comparison_slice_java_file_path = 'huawei_comparison_slicing_classifier_prob_java.txt'
# comparison_slice_python_file_path = 'huawei_comparison_slicing_classifier_prob_python.txt'
# calculate_effort(comparison_slice_java_file_path, 'java')
# print('-' * 32)
# calculate_effort(comparison_classifier_python_filepath, 'python')
# print('-' * 32)
# calculate_normalized_effort(comparison_classifier_python_filepath, 'python')
# print('-' * 32)
# calculate_normalized_effort(comparison_slice_java_file_path, 'java')
# print('-' * 32)
# calculate_normalized_effort(comparison_slice_python_file_path, 'python')
# print('-' * 32)



# prob_python_path = 'probs/prob_variant_7_finetune_1_epoch_test_python.txt'
# calculate_effort(prob_python_path, 'python')
# print('-' * 32)


def test():
    df = pd.read_csv('test_dataset_predictions.csv')
    with open('probs/huawei_pred_prob_java.csv', 'w') as file:
        writer = csv.writer(file)
        for item in df.values.tolist():
            commit_id = item[0]
            repo = item[5]
            pred_prob = item[4]
            url = repo + '/commit/' + commit_id
            pl = item[7]
            if pl == 'java':
                writer.writerow([url, pred_prob])

# test()

# prob_java_path = 'probs/prob_ensemble_classifier_test_java.txt'
# calculate_effort(prob_java_path, 'java', 0.2)
#
# print('-' * 32)
# prob_python_path = 'probs/prob_ensemble_classifier_test_python.txt'
# calculate_effort(prob_python_path, 'python', 0.2)



def calculate_prob(prob, loc):
    return (prob * (1 - min(1, math.log(loc, MAX_VULN_LOC))))


def write_new_metric(file_path, dest_path):
    df = pd.read_csv(file_path, header=None)
    url_to_label, url_to_loc_mod = get_data()
    with open(dest_path, 'w') as file:
        writer = csv.writer(file)
        for item in df.values.tolist():
            url = item[0]
            prob = item[1]
            loc = url_to_loc_mod[url]
            new_prob = calculate_prob(prob, loc)
            writer.writerow([url, new_prob])


def calculate_auc(prob_path, url_to_label, url_set=None):
    df = pd.read_csv(prob_path, header=None)

    y_test = []
    y_prob = []
    y_pred = []
    for item in df.values.tolist():
        url = item[0]
        prob = item[1]

        if url_set is not None and url not in url_set:
            continue
        
        label = url_to_label[url]

        y_test.append(label)
        y_prob.append(prob)
        if prob >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_prob)
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_prob)
    auc = metrics.roc_auc_score(y_test, y_prob)
    
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    auc_pr = metrics.average_precision_score(y_test, y_pred)
    print("auc: {}".format(auc))
    # print("auc_pr: {}".format(auc_pr))
    # print("precision: {}".format(precision))
    # print("recall: {}".format(recall))
    # print("f1: {}".format(f1))

    # for prob in y_prob:
    #     print(prob)

    # plt.plot(fpr,tpr,label="ROC curve, auc="+str(round(auc,2)))
    # plt.xlabel('True Positive Rate')
    # plt.ylabel('False Positive Rate')
    # plt.title('ROC-AUC for Python')
    # plt.legend(loc=4)
    # plt.savefig('auc_curve_python.png')
    # return auc

def test_new_metric(model_prob_path_java, model_prob_path_python, model_new_prob_java_path, model_new_prob_python_path):
    # huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    # huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'
    # model_prob_path_java = 'probs/prob_ensemble_classifier_test_java.txt'
    # model_prob_path_python = 'probs/prob_ensemble_classifier_test_python.txt'
    # huawei_new_prob_java_path = 'probs/huawei_new_prob_java.txt'
    # huawei_new_prob_python_path = 'probs/huawei_new_prob_python.txt'
    # model_new_prob_java_path = 'probs/new_prob_java.txt'
    # model_new_prob_python_path = 'probs/new_prob_python.txt'

    # write_new_metric(huawei_prob_path_java, huawei_new_prob_java_path)
    # write_new_metric(huawei_prob_path_python, huawei_new_prob_python_path)
    write_new_metric(model_prob_path_java, model_new_prob_java_path)
    write_new_metric(model_prob_path_python, model_new_prob_python_path)

    # url_to_label, url_to_loc_mod = get_data()

    # huawei_java_auc = calculate_auc(huawei_new_prob_java_path, url_to_label)
    # print("Huawei java auc: {}".format(huawei_java_auc))
    #
    # huawei_python_auc = calculate_auc(huawei_new_prob_python_path, url_to_label)
    # print("Huawei python auc: {}".format(huawei_python_auc))

    # model_java_auc = calculate_auc(model_new_prob_java_path, url_to_label)
    # print("Model java auc: {}".format(model_java_auc))

    # model_python_auc = calculate_auc(model_new_prob_python_path, url_to_label)
    # print("Model python auc: {}".format(model_python_auc))

    # print("Java effort")
    # calculate_effort(model_new_prob_java_path, 'java')
    # calculate_normalized_effort(model_new_prob_java_path, 'java')

    # print("Python effort")
    # calculate_effort(model_new_prob_python_path, 'python')
    # calculate_normalized_effort(model_new_prob_python_path, 'python')

    # print(64 * '-')

def get_pred_data(file_path, url_to_label, url_to_loc_mod):
    df = pd.read_csv(file_path, header=None)
    data = []
    for item in df.values.tolist():
        url = item[0]
        if url_to_label[url] == 0:
            continue
        data.append((url_to_loc_mod[url], item[1]))

    data = sorted(data, key=lambda x: x[1], reverse=True)

    return data


def write_csv(file_path, data):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['LOC', 'PRED_PROB'])
        for item in data:
            writer.writerow([item[0], item[1]])



def base_evaluation():
    # url_to_label, url_to_loc_mod = get_data()
    # x = -1
    # for url, loc in url_to_loc_mod.items():
    #     if url_to_label[url] == 1 and loc > x:
    #         x = loc
    #
    # print(x)
    #
    #
    # huawei_data = get_pred_data('probs/huawei_pred_prob_java.csv', url_to_label, url_to_loc_mod)
    # model_data = get_pred_data('probs/prob_ensemble_classifier_test_java.txt', url_to_label, url_to_loc_mod)
    # write_csv('huawei_pos_loc.csv', huawei_data)
    # write_csv('model_pos_loc.csv', model_data)


    # calculate_effort('probs/prob_ensemble_classifier_test_java.txt', 'java')
    # calculate_effort('probs/prob_ensemble_classifier_test_java.txt', 'java', 0.02)


    # calculate_normalized_effort('probs/new_prob_java.txt', 'java')

    # df = pd.read_csv('huawei_popt.csv', header=None)
    # x_huawei = []
    # y_huawei = []
    # for item in df.values.tolist():
    #     x_huawei.append(item[0])
    #     y_huawei.append(item[1])
    #
    # df = pd.read_csv('model_popt.csv', header=None)
    # x_model = []
    # y_model = []
    # for item in df.values.tolist():
    #     x_model.append(item[0])
    #     y_model.append(item[1])
    #
    # plt.plot(x_huawei, y_huawei, label='Huawei')
    # plt.plot(x_model, y_model, label='Our model')
    # plt.title("Comparison on cost effort as inspected is increased")
    # plt.xlabel("% Inspected LOC")
    # plt.ylabel("Cost-Effort")
    # plt.legend()
    # plt.show()
    #
    # calculate_effort('probs/huawei_pred_prob_python.csv', 'python')
    # calculate_normalized_effort('probs/huawei_pred_prob_python.csv', 'python')

    # url_to_label, url_to_loc_mod = get_data()
    # print(calculate_auc('probs/prob_ensemble_classifier_test_python.txt', url_to_label))

    # print("Variant 1...")
    # calculate_effort('probs/prob_variant_1_finetune_1_epoch_test_python.txt', 'python')
    #
    # print("Variant 2...")
    # calculate_effort('probs/prob_variant_2_finetune_1_epoch_test_python.txt', 'python')
    #
    # print("Variant 3...")
    # calculate_effort('probs/prob_variant_3_finetune_1_epoch_test_python.txt', 'python')
    #
    # print("Variant 5...")
    # calculate_effort('probs/prob_variant_5_finetune_1_epoch_test_python.txt', 'python')
    #
    # print("Variant 6...")
    # calculate_effort('probs/prob_variant_6_finetune_1_epoch_test_python.txt', 'python')
    #
    # print("Variant 7...")
    # calculate_effort('probs/prob_variant_7_finetune_1_epoch_test_python.txt', 'python')
    #
    # print("Variant 8...")
    # calculate_effort('probs/prob_variant_8_finetune_1_epoch_test_python.txt', 'python')


    # java_predicted_paths = [
    #     'probs/prob_variant_1_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    #     'probs/prob_variant_2_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    #     'probs/prob_variant_3_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    #     'probs/prob_variant_5_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    #     'probs/prob_variant_6_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    #     'probs/prob_variant_7_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    #     'probs/prob_variant_8_finetune_1_epoch_test_java.txt_predicted_indices.txt',
    # ]
    #
    # python_predicted_paths = [
    #     'probs/prob_variant_1_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    #     'probs/prob_variant_2_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    #     'probs/prob_variant_3_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    #     'probs/prob_variant_5_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    #     'probs/prob_variant_6_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    #     'probs/prob_variant_7_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    #     'probs/prob_variant_8_finetune_1_epoch_test_python.txt_predicted_indices.txt',
    # ]
    #
    # vuln_set = set()
    # for path in python_predicted_paths:
    #     with open(path, 'r') as file:
    #         lines = file.read().splitlines()
    #         for line in lines:
    #             vuln_set.add(line)
    # print(len(vuln_set))

    # model_prob_path_java = 'probs/prob_ablation_1_java.txt'
    # model_prob_path_python = 'probs/prob_ablation_1_python.txt'
    # model_new_prob_java_path = 'probs/prob_ablation_1_java_new.txt'
    # model_new_prob_python_path = 'probs/prob_ablation_1_python_new.txt'


    # calculate_effort('probs/new_prob_python.txt', 'python')

    huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'

    # calculate_effort(huawei_prob_path_python, 'python')

    cnn_model_prob_path_java = 'probs/prob_ensemble_classifier_file_level_cnn_test_java.txt'
    cnn_model_prob_path_python = 'probs/prob_ensemble_classifier_file_level_cnn_test_python.txt'
    cnn_model_prob_path_java_new = 'probs/prob_ensemble_classifier_file_level_cnn_test_java_new.txt'
    cnn_model_prob_path_python_new = 'probs/prob_ensemble_classifier_file_level_cnn_test_python_new.txt'

    # test_new_metric(cnn_model_prob_path_java, cnn_model_prob_path_python, cnn_model_prob_path_java_new, cnn_model_prob_path_python_new)

    # calculate_normalized_effort(cnn_model_prob_path_python_new, 'python')

    url_to_label, url_to_loc_mod = get_data()

    model_prob_path_java = 'probs/prob_ensemble_classifier_test_java.txt'
    model_prob_path_python = 'probs/prob_ensemble_classifier_test_python.txt'
    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'
    # calculate_auc(model_new_prob_python_path, url_to_label)


    calculate_effort(model_prob_path_java, 'java')

    # vulcurator_r28_java_prob_path = 'probs/r28_7_new_prob_java.txt'
    # vulcurator_r28_python_prob_path = 'probs/r28_7_new_prob_python.txt'

    # huawei_r28_prob_path_java = 'probs/r28_7_huawei_pred_prob_java.csv'
    # huawei_r28_prob_path_python = 'probs/r28_7_huawei_pred_prob_python.csv'

    # model_java_auc = calculate_auc(huawei_r28_prob_path_python, url_to_label)
    # print("Model auc: {}".format(model_java_auc))

    # url_to_hunk_count = get_url_to_hunk_count()
    # calculate_effort(huawei_r28_prob_path_python, 'python', url_to_hunk_count)
    # calculate_normalized_effort(huawei_r28_prob_path_python, 'python', url_to_hunk_count)



def evaluate_security_subdataset():
    global dataset_name
    dataset_name = sec_dataset_name
    
    url_to_label, url_to_loc_mod, java_url_set, python_url_set = get_security_data(sec_dataset_name)

    model_new_prob_java_path = 'probs/new_prob_java.txt'
    model_new_prob_python_path = 'probs/new_prob_python.txt'

    huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'

    # calculate_auc(model_new_prob_java_path, url_to_label, java_url_set)
    # print('-'*32)
    # calculate_auc(model_new_prob_python_path, url_to_label, python_url_set)
    # print('-'*32)
    # calculate_auc(huawei_prob_path_java, url_to_label, java_url_set)
    # print('-'*32)
    # calculate_auc(huawei_prob_path_python, url_to_label, python_url_set)

    # calculate_effort(model_new_prob_java_path, 'java', url_set=java_url_set)
    # print('-'*32)
    # calculate_effort(model_new_prob_python_path, 'python', url_set=python_url_set)
    # print('-'*32)
    # calculate_effort(huawei_prob_path_java, 'java', url_set=java_url_set)
    # print('-'*32)
    # calculate_effort(huawei_prob_path_python, 'python', url_set=python_url_set)

    # calculate_normalized_effort(model_new_prob_java_path, 'java', url_set=java_url_set)
    # print('-'*32)
    # calculate_normalized_effort(model_new_prob_python_path, 'python', url_set=python_url_set)
    # print('-'*32)
    # calculate_normalized_effort(huawei_prob_path_java, 'java', url_set=java_url_set)
    # print('-'*32)
    # calculate_normalized_effort(huawei_prob_path_python, 'python', url_set=python_url_set)


def evaluate_different_design():
    # model_line_lstm_prob_path_java = 'probs/prob_ensemble_classifier_line_lstm_test_java.txt'
    # model_line_lstm_prob_path_python = 'probs/prob_ensemble_classifier_line_lstm_test_python.txt'
    # model_line_lstm_prob_path_java_new = 'probs/prob_ensemble_classifier_line_lstm_test_java_new.txt'
    # model_line_lstm_prob_path_python_new = 'probs/prob_ensemble_classifier_line_lstm_test_python_new.txt'

    # # test_new_metric(model_line_lstm_prob_path_java, model_line_lstm_prob_path_python, model_line_lstm_prob_path_java_new, model_line_lstm_prob_path_python_new)
    
    # url_to_label, url_to_loc_mod = get_data()

    # # calculate_auc(model_line_lstm_prob_path_python_new, url_to_label)
    # calculate_effort(model_line_lstm_prob_path_python_new, 'python')
    # calculate_normalized_effort(model_line_lstm_prob_path_python_new, 'python')


    # model_line_gru_prob_path_java = 'probs/prob_ensemble_classifier_line_gru_test_java.txt'
    # model_line_gru_prob_path_python = 'probs/prob_ensemble_classifier_line_gru_test_python.txt'
    # model_line_gru_prob_path_java_new = 'probs/prob_ensemble_classifier_line_gru_test_java_new.txt'
    # model_line_gru_prob_path_python_new = 'probs/prob_ensemble_classifier_line_gru_test_python_new.txt'

    # test_new_metric(model_line_gru_prob_path_java, model_line_gru_prob_path_python, model_line_gru_prob_path_java_new, model_line_gru_prob_path_python_new)
    
    # url_to_label, url_to_loc_mod = get_data()

    # calculate_auc(model_line_gru_prob_path_java_new, url_to_label)
    # calculate_effort(model_line_gru_prob_path_java_new, 'java')
    # calculate_normalized_effort(model_line_gru_prob_path_java_new, 'java')

    
    model_hunk_fcn_prob_path_java = 'probs/prob_ensemble_classifier_hunk_fcn_test_java.txt'
    model_hunk_fcn_prob_path_python = 'probs/prob_ensemble_classifier_hunk_fcn_test_python.txt'
    model_hunk_fcn_prob_path_java_new = 'probs/prob_ensemble_classifier_hunk_fcn_test_java_new.txt'
    model_hunk_fcn_prob_path_python_new = 'probs/prob_ensemble_classifier_hunk_fcn_test_python_new.txt'

    test_new_metric(model_hunk_fcn_prob_path_java, model_hunk_fcn_prob_path_python, model_hunk_fcn_prob_path_java_new, model_hunk_fcn_prob_path_python_new)
    
    url_to_label, url_to_loc_mod = get_data()

    calculate_auc(model_hunk_fcn_prob_path_python_new, url_to_label)
    calculate_effort(model_hunk_fcn_prob_path_python_new, 'python')
    calculate_normalized_effort(model_hunk_fcn_prob_path_python_new, 'python')


def evaluate_pca():

    model_pca_prob_path_java = 'probs/prob_ensemble_classifier_pca_070_test_java.txt'
    model_pca_prob_path_python = 'probs/prob_ensemble_classifier_pca_070_test_python.txt'
    model_pca_prob_path_java_new = 'probs/prob_ensemble_classifier_pca_070_test_java_new.txt'
    model_pca_prob_path_python_new = 'probs/prob_ensemble_classifier_pca_070_test_python_new.txt'

    test_new_metric(model_pca_prob_path_java, model_pca_prob_path_python, model_pca_prob_path_java_new, model_pca_prob_path_python_new)
    
    url_to_label, url_to_loc_mod = get_data()

    calculate_auc(model_pca_prob_path_java_new, url_to_label)
    calculate_effort(model_pca_prob_path_java_new, 'java')
    calculate_normalized_effort(model_pca_prob_path_java_new, 'java')

def evaluate_cfs():
    model_prob_path_java = 'probs/prob_ensemble_classifier_cfs_test_java.txt'
    model_prob_path_python = 'probs/prob_ensemble_classifier_cfs_test_python.txt'
    model_prob_path_java_new = 'probs/prob_ensemble_classifier_cfs_test_java_new.txt'
    model_prob_path_python_new = 'probs/prob_ensemble_classifier_cfs_test_python_new.txt'

    test_new_metric(model_prob_path_java, model_prob_path_python, model_prob_path_java_new, model_prob_path_python_new)
    
    url_to_label, url_to_loc_mod = get_data()

    calculate_auc(model_prob_path_python_new, url_to_label)
    calculate_effort(model_prob_path_python_new, 'python')
    calculate_normalized_effort(model_prob_path_python_new, 'python')


if __name__ == '__main__':
    # evaluate_security_subdataset()
    # evaluate_different_design()
    evaluate_cfs()