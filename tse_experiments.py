from cgitb import small
import pandas as pd
from sklearn import metrics
from torch import threshold
import utils
import matplotlib.pyplot as plt
import csv 
from tqdm import tqdm
import lime_variant_visualizer
import metrices_calculator
import numpy as np

BIG_COMMIT_THRESHOLD = 512
SMALL_COMMIT_THRESHOLD = 50
# HUNK_COUNT_THRESHOLD = 5

dataset_name = 'ase_dataset_sept_19_2021.csv'

def read_url_to_token_count():
    url_to_added_count = {}
    url_to_removed_count = {}

    big_commit = set()
    small_commit = set()

    tokens = []

    df = pd.read_csv('huawei_dataset_url_to_token_count.csv')

    for item in df.values.tolist():
        added_count = item[2]
        removed_count = item[3]
        url_to_added_count[item[1]] = added_count
        url_to_removed_count[item[1]] = removed_count

        token_count = added_count + removed_count

        if token_count > BIG_COMMIT_THRESHOLD:
            big_commit.add(item[1])
        if token_count < SMALL_COMMIT_THRESHOLD:
            small_commit.add(item[1])

    # fig = plt.figure(figsize =(10, 7))

    # plt.boxplot(tokens, showfliers=False)
    # plt.savefig('token_distribution.jpg')

    return url_to_added_count, url_to_removed_count, big_commit, small_commit


def count_index(index_path, big_commit, small_commit):
    with open(index_path, 'r') as file:
        lines = file.readlines()
        predicted_commit = [line.rstrip() for line in lines]

    big_count, small_count = 0, 0

    for commit in predicted_commit:
        if commit in big_commit:
            big_count += 1
        if commit in small_commit:
            small_count += 1

    return big_count, small_count

def count_result_based_on_token():
    url_to_add_count, url_to_removed_count, big_commit, small_commit = read_url_to_token_count()

    url_data, label_data, url_to_pl, url_to_label = utils.get_data('ase_dataset_sept_19_2021.csv', need_pl=True)

    # giang, only calculate big vulnerability-fixing commits

    big_commit_java_test = set()
    small_commit_java_test = set()

    count = 0
    for i, url in enumerate(url_data['test_python']):
        if label_data['test_python'][i] == 1:
            count += 1
            if url in big_commit:
                big_commit_java_test.add(url)
            if url in small_commit:
                small_commit_java_test.add(url)


    effort_big_count, effort_small_count = count_index('probs/huawei_pred_prob_python.csv_predicted_indices_cost_effort_20.txt', big_commit, small_commit)
    # effort_big_count, effort_small_count = count_index('probs/new_prob_python.txt_predicted_indices_cost_effort_15.txt', big_commit, small_commit)

    print(len(big_commit_java_test))
    print(len(small_commit_java_test))
    print(effort_big_count)
    print(effort_small_count)
    
    print(count)


def qualitative_analysis():
    vulcurator_java_indices_path = 'probs/new_prob_java.txt_predicted_indices_cost_effort_20.txt'
    vulcurator_python_indices_path = 'probs/new_prob_python.txt_predicted_indices_cost_effort_20.txt'
    huawei_java_indices_path = 'probs/huawei_pred_prob_java.csv_predicted_indices_cost_effort_20.txt'
    huawei_python_indices_path = 'probs/huawei_pred_prob_python.csv_predicted_indices_cost_effort_20.txt'

    with open(vulcurator_java_indices_path) as file:
        lines = file.readlines()
        vulcurator_java_indices = set(line.rstrip() for line in lines)

    with open(vulcurator_python_indices_path) as file:
        lines = file.readlines()
        vulcurator_python_indices = set(line.rstrip() for line in lines)

    with open(huawei_java_indices_path) as file:
        lines = file.readlines()
        huawei_java_indices = set(line.rstrip() for line in lines)

    with open(huawei_python_indices_path) as file:
        lines = file.readlines()
        huawei_python_indices = set(line.rstrip() for line in lines)  

    java_vul_set = vulcurator_java_indices - huawei_java_indices
    python_vul_set = vulcurator_python_indices - huawei_python_indices
    # print(len(vulcurator_java_indices))
    # print(len(huawei_java_indices))
    # print(len(vulcurator_java_indices - huawei_java_indices))

    # for url in java_vul_set:
    #     print('https://github.com/' + url)

    print(len(vulcurator_python_indices))
    print(len(huawei_python_indices))
    print(len(python_vul_set))
    
    for url in python_vul_set:
        print('https://github.com/' + url)


def plot_hunk_count():
    df = pd.read_csv('huawei_dataset_url_to_hunk_count.csv')

    java_count = []
    python_count = []
    
    for item in df.to_numpy().tolist():
        url = item[0]
        hunk_count = item[1]
        label = item[2]
        partition = item[3]
        pl = item[4]

        if label == 1 and partition == 'test' and hunk_count > 0 and hunk_count <= 3:
            if pl == 'java':
                java_count.append(hunk_count)
            else:
                python_count.append(hunk_count)

    print(len(java_count))
    print(len(python_count))
    plt.boxplot(python_count, showfliers=False)
    plt.savefig('hunk_count.png')


def write_r28_probs(old_probs_path, new_probs_path, url_to_hunk_count):
     with open(old_probs_path, 'r') as file1:
        reader = csv.reader(file1)
        with open(new_probs_path, 'w') as file2:
            writer = csv.writer(file2)
            for row in reader:
                url = row[0]
                prob = float(row[1])
                hunk_count = url_to_hunk_count[url]
                if hunk_count >= 7:
                    writer.writerow([url, prob])


def get_url_to_hunk_count():
    url_to_hunk_count = {}
    df = pd.read_csv('huawei_dataset_url_to_hunk_count.csv')

    
    for item in df.to_numpy().tolist():
        url = item[0]
        hunk_count = item[1]
        label = item[2]
        partition = item[3]
        pl = item[4]

        url_to_hunk_count[url] = hunk_count
    
    return url_to_hunk_count

    
def write_all_r28_probs():
    url_to_hunk_count = get_url_to_hunk_count()

    vulcurator_java_prob_path = 'probs/new_prob_java.txt'
    vulcurator_python_prob_path = 'probs/new_prob_python.txt'

    huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'

    vulcurator_r28_java_prob_path = 'probs/r28_7_new_prob_java.txt'
    vulcurator_r28_python_prob_path = 'probs/r28_7_new_prob_python.txt'

    huawei_r28_prob_path_java = 'probs/r28_7_huawei_pred_prob_java.csv'
    huawei_r28_prob_path_python = 'probs/r28_7_huawei_pred_prob_python.csv'

    write_r28_probs(vulcurator_java_prob_path, vulcurator_r28_java_prob_path, url_to_hunk_count)
    write_r28_probs(vulcurator_python_prob_path, vulcurator_r28_python_prob_path, url_to_hunk_count)
    write_r28_probs(huawei_prob_path_java, huawei_r28_prob_path_java, url_to_hunk_count)
    write_r28_probs(huawei_prob_path_python, huawei_r28_prob_path_python, url_to_hunk_count)


def read_probs_from_file(file_path):
    url_to_probs = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            url_to_probs[row[0]] = float(row[1])

    return url_to_probs


def find_threshold(url_to_val_probs, url_to_label, url_to_pl, lang):
    val_y_prob = []
    val_y_test = []
    for url, prob in url_to_val_probs.items():
        if url_to_pl[url] != lang:
            continue

        val_y_prob.append(prob)
        val_y_test.append(url_to_label[url])

    best_threshold = -1
    best_f1 = 0
    for prob1 in tqdm(val_y_prob):
        val_y_pred = []
        for prob2 in val_y_prob:
            if prob2 >= prob1:
                val_y_pred.append(1)
            else:
                val_y_pred.append(0)
        f1 = metrics.f1_score(y_true=val_y_test, y_pred=val_y_pred)
        if f1 > best_f1:
            print("New Best F1: {}".format(f1))
            best_f1 = f1
            best_threshold = prob1

    auc = metrics.roc_auc_score(val_y_test, val_y_prob)
    print("auc: {}".format(auc))
    return best_threshold

def calculate_metrics(url_to_probs, url_to_label, threshold):
    y_pred = []
    y_test = []
    for url, prob in url_to_probs.items():
        if prob >= threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

        y_test.append(url_to_label[url])
    
    precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
    recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
    f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
    tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pred, y_true=y_test).ravel()
    fpr = fp/(fp + tn)
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1: {}".format(f1))
    print("fpr: {}".format(fpr))
    print(metrics.confusion_matrix(y_test, y_pred))

def r27_finetune_threshold():
    val_probs_path = 'probs/prob_ensemble_classifier_val.txt'
    test_java_probs_path = 'probs/new_prob_java.txt'
    test_python_probs_path = 'probs/new_prob_python.txt'

    huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'

    url_to_val_probs = read_probs_from_file(val_probs_path)
    url_to_test_java_probs = read_probs_from_file(huawei_prob_path_java)
    url_to_test_python_probs = read_probs_from_file(huawei_prob_path_python)

    url_data, label_data, url_to_pl, url_to_label = utils.get_data(dataset_name, need_pl=True)


    java_threshold = find_threshold(url_to_test_java_probs, url_to_label, url_to_pl, 'java')
    print("Threshold: {}".format(java_threshold))
    calculate_metrics(url_to_test_java_probs, url_to_label, java_threshold)


def grid_search(url_to_val_probs, url_to_label, url_to_pl, lang):
    val_y_prob = []
    val_y_test = []
    for url, prob in url_to_val_probs.items():
        if url_to_pl[url] != lang:
            continue

        val_y_prob.append(prob)
        val_y_test.append(url_to_label[url])

    current_threshold = 0
    best_threshold = -1
    best_f1 = 0
    
    for i in tqdm(range(0, 10000)):
        current_threshold = 0.0001 * i
        val_y_pred = []
        for prob2 in val_y_prob:
            if prob2 >= current_threshold:
                val_y_pred.append(1)
            else:
                val_y_pred.append(0)
        f1 = metrics.f1_score(y_true=val_y_test, y_pred=val_y_pred)
        if f1 > best_f1:
            print("New Best F1: {}, threshold: {}".format(f1, current_threshold))
            best_f1 = f1
            best_threshold = current_threshold

    auc = metrics.roc_auc_score(val_y_test, val_y_prob)
    print("auc: {}".format(auc))
    return best_threshold

def r27_grid_search():
    val_probs_path = 'probs/prob_ensemble_classifier_val.txt'
    test_java_probs_path = 'probs/new_prob_java.txt'
    test_python_probs_path = 'probs/new_prob_python.txt'

    huawei_prob_path_java = 'probs/huawei_pred_prob_java.csv'
    huawei_prob_path_python = 'probs/huawei_pred_prob_python.csv'

    url_to_val_probs = read_probs_from_file(val_probs_path)
    url_to_test_java_probs = read_probs_from_file(huawei_prob_path_java)
    url_to_test_python_probs = read_probs_from_file(huawei_prob_path_python)

    url_data, label_data, url_to_pl, url_to_label = utils.get_data(dataset_name, need_pl=True)


    threshold = grid_search(url_to_test_java_probs, url_to_label, url_to_pl, 'java')
    print("Threshold: {}".format(threshold))
    calculate_metrics(url_to_test_java_probs, url_to_label, threshold)


def get_word_set(top_k):
    words, frequencies = zip(*top_k)

    return set(words)


def check_overlap(list_top_tokens):
    for i, top_i in enumerate(list_top_tokens):
        for j, top_j in enumerate(list_top_tokens):
            if i != j and top_i == top_j:
                return True
    
    return False


def r33_feature_understanding():
    
    top_tokens_1 = lime_variant_visualizer.visualize('variant_1_explanation.json')
    top_tokens_2 = lime_variant_visualizer.visualize('variant_2_explanation.json')
    top_tokens_3 = lime_variant_visualizer.visualize('variant_3_explanation.json')
    top_tokens_5 = lime_variant_visualizer.visualize('variant_5_explanation.json')
    top_tokens_6 = lime_variant_visualizer.visualize('variant_6_explanation.json')
    top_tokens_7 = lime_variant_visualizer.visualize('variant_7_explanation.json')
    top_tokens_8 = lime_variant_visualizer.visualize('variant_8_explanation.json')

    top_1_set = get_word_set(top_tokens_1)
    top_2_set = get_word_set(top_tokens_2)
    top_3_set = get_word_set(top_tokens_3)
    top_5_set = get_word_set(top_tokens_5)
    top_6_set = get_word_set(top_tokens_6)
    top_7_set = get_word_set(top_tokens_7)
    top_8_set = get_word_set(top_tokens_8)

    # union without 1. Similar to others
    union_1 = set().union(*[top_2_set, top_3_set, top_5_set, top_6_set, top_7_set, top_8_set])
    intersect_1 = top_1_set - union_1

    print(intersect_1)


    union_2 = set().union(*[top_1_set, top_3_set, top_5_set, top_6_set, top_7_set, top_8_set])
    intersect_2 = top_2_set - union_2

    print(intersect_2)


    union_3 = set().union(*[top_1_set, top_2_set, top_5_set, top_6_set, top_7_set, top_8_set])
    intersect_3 = top_3_set - union_3

    print(intersect_3)


    union_5 = set().union(*[top_1_set, top_2_set, top_3_set, top_6_set, top_7_set, top_8_set])
    intersect_5 = top_5_set - union_5

    print(intersect_5)


    union_6 = set().union(*[top_1_set, top_2_set, top_3_set, top_5_set, top_7_set, top_8_set])
    intersect_6 = top_6_set - union_6

    print(intersect_6)


    union_7 = set().union(*[top_1_set, top_2_set, top_3_set, top_5_set, top_6_set, top_8_set])
    intersect_7 = top_7_set - union_7

    print(intersect_7)


    union_8 = set().union(*[top_1_set, top_2_set, top_3_set, top_5_set, top_6_set, top_7_set])
    intersect_8 = top_8_set - union_8

    print(intersect_8)

    has_overlap = check_overlap([top_1_set, top_2_set, top_3_set, top_5_set, top_6_set, top_7_set, top_8_set])

    print("Has overlap: {}".format(has_overlap))


    tops = [top_1_set, top_2_set, top_3_set, top_5_set, top_6_set, top_7_set, top_8_set]

    overlap_to_count = {}

    for i, x in enumerate(tops):
        for j, y in enumerate(tops):
            if i > j:
                overlap = len(x.intersection(y))
                if overlap not in overlap_to_count:
                    overlap_to_count[overlap] = 0
                
                overlap_to_count[overlap] += 1
    
    for key, value in overlap_to_count.items():
        print("Key: {}, Value: {}".format(key, value))


def r32_distribution():
    print("Reading dataset")
    df = pd.read_csv(dataset_name)
    df = df[df.partition == 'test']
    df_java_test = df[df.PL == 'java']
    df_python_test = df[df.PL == 'python']

    df_java_test = df_java_test[['repo', 'commit_id', 'LOC_MOD']]
    df_python_test = df_python_test[['repo', 'commit_id', 'LOC_MOD']]

    java_items = df_java_test.values.tolist()
    python_items = df_python_test.values.tolist()

    java_url_to_loc = metrices_calculator.get_url_to_loc(java_items)
    python_url_to_loc = metrices_calculator.get_url_to_loc(python_items)

    java_locs = list(java_url_to_loc.values())
    python_locs = list(python_url_to_loc.values())


    print("Java std: {}".format(np.std(java_locs)))
    print("Python std: {}".format(np.std(python_locs)))

    plt.boxplot(java_locs, showfliers=False)
    plt.savefig('img/r32_test_java_loc.jpg')
    plt.close()

    plt.boxplot(python_locs, showfliers=False)
    plt.savefig('img/r32_test_python_loc.jpg')
    plt.close()

if __name__ == '__main__':
    # count_result_based_on_token()
    # qualitative_analysis()
    # plot_hunk_count()
    # write_all_r28_probs()

    # r27_grid_search()

    r33_feature_understanding()

    # r32_distribution()