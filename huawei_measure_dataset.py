from _csv import Error

import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics

from transformers import RobertaTokenizer

from sklearn.linear_model import LogisticRegression
import preprocess_variant_3

dataset_name = 'ase_dataset_sept_19_2021.csv'

directory = os.path.dirname(os.path.abspath(__file__))
slicing_result_folder_path = os.path.join(directory, '../slicing_result_30092021')

slicing_structured_result_folder_path = os.path.join(directory, '../slicing_structured_result_30092021')
slicing_outputs_folder_path = os.path.join(directory, '../slicing_output_context_only_03102021')

accept_url_set = set()


def get_loc_without_comments(diff, added_version):
    count = 0
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '*', '*/', '#')):
                continue
            count += 1

    return count


def get_loc_statistics(df):
    items = df.to_numpy().tolist()

    url_to_items = {}

    for index, item in enumerate(items):
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id

        if url not in url_to_items:
            url_to_items[url] = []

        url_to_items[url].append((index, item))

    url_to_added_loc = {}
    url_to_removed_loc = {}
    for url, items in url_to_items.items():
        url_to_added_loc[url] = 0
        url_to_removed_loc[url] = 0
        for index, item in items:
            diff = item[3]
            url_to_added_loc[url] += get_loc_without_comments(diff, True)
            url_to_removed_loc[url] += get_loc_without_comments(diff, False)

    added_loc_list = []
    removed_loc_list = []
    for url in url_to_added_loc.keys():
        added_loc_list.append(url_to_added_loc[url])
        removed_loc_list.append(url_to_removed_loc[url])

    return added_loc_list, removed_loc_list


def write_loc_to_file(file_path, loc_list):
    with open(file_path, 'w') as writer:
        for loc in loc_list:
            writer.write(str(loc) + '\n')


def write_loc_statistics_to_file():
    url_to_index = {}
    with open('url_to_index.txt', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            url_to_index[row[0]] = row[1]

    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[df.PL == 'java']
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL']]
    pos_df = df[df.label == 1]
    neg_df = df[df.label == 0]

    pos_added_loc_list, pos_removed_loc_list = get_loc_statistics(pos_df)
    neg_added_loc_list, neg_removed_loc_list = get_loc_statistics(neg_df)

    write_loc_to_file('pos_java_added_loc_list.txt', pos_added_loc_list)
    write_loc_to_file('pos_java_removed_loc_list.txt', pos_removed_loc_list)
    write_loc_to_file('neg_java_added_loc_list.txt', neg_added_loc_list)
    write_loc_to_file('neg_java_removed_loc_list.txt', neg_removed_loc_list)


def read_loc_list(file_path):
    loc_list = []
    with open(file_path, 'r') as reader:
        lines = reader.read().split('\n')[:-1]
        for line in lines:
            loc_list.append(int(line))

    return loc_list


def get_slice_percentage(loc_list):
    loc_list.sort()
    LINE_LIMIT = 24

    zero_index = 0
    for index, value in enumerate(loc_list):
        if value > 0:
            zero_index = index
            break

    for index, value in enumerate(loc_list):
        if value == LINE_LIMIT:
            return (index - zero_index)/len(loc_list)


def analyze_loc_statistics():
    pos_added_loc_list = read_loc_list('pos_added_loc_list.txt')
    pos_removed_loc_list = read_loc_list('pos_removed_loc_list.txt')
    neg_added_loc_list = read_loc_list('neg_added_loc_list.txt')
    neg_removed_loc_list = read_loc_list('neg_removed_loc_list.txt')

    print("VF added slice-able: {}".format(get_slice_percentage(pos_added_loc_list)))
    print("VF removed slice-able: {}".format(get_slice_percentage(pos_removed_loc_list)))
    print("NVF added slice-able: {}".format(get_slice_percentage(neg_added_loc_list)))
    print("NVF removed slice-able: {}".format(get_slice_percentage(neg_removed_loc_list)))

    plt.boxplot([pos_added_loc_list, pos_removed_loc_list, neg_added_loc_list, neg_removed_loc_list], labels= ['VF_added_loc', 'VF_removed_loc', 'NVF_added_loc', 'NVF_removed_loc'], showfliers=False)
    plt.title("Distribution in number of lines of code for both Java and Python")
    plt.ylabel("No.Lines of code")
    plt.show()


def get_loc(diff, added_version):
    loc = 0
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line == '' or line.startswith(('//', '/*', '/**', '*', '*/', '#')):
                continue
            loc += 1

    return loc


def filter_fn(row):
    global accept_url_set

    url = row['repo'] + '/commit/' + row['commit_id']
    if url in accept_url_set:
        return True
    else:
        return False


def extract_subdataset():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    print(len(df))
    items = df.values.tolist()

    url_to_added_loc = {}
    url_to_deleted_loc = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        diff = item[6]
        added_loc = get_loc(diff, True)
        deleted_loc = get_loc(diff, False)
        if url not in url_to_added_loc:
            url_to_added_loc[url] = 0
        if url not in url_to_deleted_loc:
            url_to_deleted_loc[url] = 0

        url_to_added_loc[url] += added_loc
        url_to_deleted_loc[url] += deleted_loc

    accept_url_set = set()
    for url in url_to_added_loc.keys():
        added_loc = url_to_added_loc[url]
        deleted_loc = url_to_deleted_loc[url]
        if added_loc <= 15 and deleted_loc <= 15:
            accept_url_set.add(url)

    new_items = []
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        if url in accept_url_set:
            new_items.append(item)

    # new_df = pd.DataFrame(new_items, columns=df.columns.values.tolist())
    with open('huawei_csv_subset.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(df.columns.values.tolist())
        for item in new_items:
            writer.writerow(item)
    # new_df.to_csv('huawei_sub_dataset_test_slicing_edited.csv')


def analyze_slicing_result():
    index_to_url = {}
    with open('url_to_index.txt', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index_to_url[int(row[1])] = row[0]

    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'label']]
    items = df.to_numpy().tolist()

    url_to_label = {}
    for index, item in enumerate(items):
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id

        url_to_label[url] = item[2]


    total_index_set = set()
    pos_index_set = set()
    neg_index_set = set()
    remaining_lines = {}
    pos_remaining_lines = {}
    neg_remaining_lines = {}
    print("Calculating remaining lines...")
    for file_name in os.listdir(slicing_structured_result_folder_path):
        file_path = slicing_structured_result_folder_path + '/' + file_name
        index = int(file_name.split('_')[0])
        label = url_to_label[index_to_url[index]]
        try:
            enhanced_count = 0
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) != 4:
                        continue
                    if row[3] == 'enhanced':
                       enhanced_count += 1

                output_file_path = slicing_outputs_folder_path + '/' + file_name

                with open(output_file_path, 'r') as reader:
                    code_len = len(reader.read().split('\n')) - 1

            total_index_set.add(index)
            if label == 1:
                pos_index_set.add(index)
            else:
                neg_index_set.add(index)

            if enhanced_count > code_len:
                sub = enhanced_count - code_len
                if index not in remaining_lines:
                    remaining_lines[index] = 0
                remaining_lines[index] += sub

                if label == 1:
                    if index not in pos_remaining_lines:
                        pos_remaining_lines[index] = 0
                    pos_remaining_lines[index] += sub
                else:
                    if index not in neg_remaining_lines:
                        neg_remaining_lines[index] = 0
                    neg_remaining_lines[index] += sub

        except Error:
            continue

    print("Percentage of total commit fully used: {}/{} => {}".format(len(remaining_lines.keys()), len(total_index_set), len(remaining_lines.keys()) / len(total_index_set)))
    print("Percentage of VF commit fully used: {}/{} => {}".format(len(pos_remaining_lines.keys()), len(pos_index_set), len(pos_remaining_lines.keys()) / len(pos_index_set)))
    print("Percentage of NVF commit fully used: {}/{} => {}".format(len(neg_remaining_lines.keys()), len(total_index_set), len(neg_remaining_lines.keys()) / len(neg_index_set)))

    total_remain_list = list(remaining_lines.values())
    pos_remaining_list = list(pos_remaining_lines.values())
    neg_remaining_list = list(neg_remaining_lines.values())

    plt.boxplot([total_remain_list, pos_remaining_list, neg_remaining_list],
                labels=['All commits', "VF commits", "NVF commits"], showfliers=False)
    plt.title("Distribution in number of remaining lines after presenting slicing results")
    plt.ylabel("No.Lines of code")
    plt.savefig("remaining_lines.png")


def ensemble_result():
    comparison_classifier_python_filepath_new = 'huawei_comparison_classifier_29092021_prob_python.txt'
    slicing_context_python_file_path = 'huawei_context_slicing_classifier_prob_python.txt'
    index_to_label = {}
    comparison_index_to_pred = {}
    slice_index_to_pred = {}

    with open(comparison_classifier_python_filepath_new, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index = row[0]
            prob = row[1]
            y_test = row[3]

            index_to_label[index] = y_test
            comparison_index_to_pred[index] = prob

    with open(slicing_context_python_file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index = row[0]
            prob = row[1]
            y_test = row[3]

            index_to_label[index] = y_test
            slice_index_to_pred[index] = prob

    index_to_final_prob = {}


def get_sub_dataset():
    df = pd.read_csv(dataset_name)
    df = df[(df.partition == 'test') & (df.label == 1)]
    df = df[['commit_id', 'repo']]
    items = df.values.tolist()
    with open('huawei_pos_test.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['commit_url'])
        url_set = set()
        for item in items:
            url_set.add('https://github.com/' + item[1] + '/commit/' + item[0])

        for url in url_set:
            writer.writerow([url])


def get_url_to_prediction(prob_path, lang):
    url_to_prediction = {}

    patch_train, patch_test, patch_validation, label_train, label_test, label_validation, test_index_to_loc_mod, test_index_to_label, test_index_to_url \
        = huawei_pure_classifier.get_data(lang=lang)
    with open(prob_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index = int(row[0])
            prob = float(row[1])
            url = test_index_to_url[index]
            url_to_prediction[url] = prob

    return url_to_prediction


def add_column():
    df = pd.read_csv('huawei_pos_test_edited.csv')
    url_list = df['commit_url'].values.tolist()
    url_to_predicted = {}
    java_prob_path = 'huawei_pure_classifier_prob_java.txt'
    python_prob_path = 'huawei_pure_classifier_prob_python.txt'

    d1 = get_url_to_prediction(java_prob_path, 'java')
    d2 = get_url_to_prediction(python_prob_path, 'python')
    url_to_predicted.update(d1)
    url_to_predicted.update(d2)

    prob_list = []
    for url in url_list:
        short_url = url[len('https://github.com/'):]
        prob = url_to_predicted[short_url] * 100
        prob_list.append(float("{0:.4f}".format(prob)))

    plt.boxplot([prob_list],
                labels=[''], showfliers=True)
    plt.title("Predicted probabilities for vulnerability-fixing commit")
    plt.ylabel("Percentage")
    plt.show()

    plt.hist(prob_list, bins=100)
    plt.title("Predicted probabilities for vulnerability-fixing commit")
    plt.ylabel("Percentage")
    plt.show()


def do_something():
    huawei_test_pred_file = 'test_dataset_predictions.csv'
    df = pd.read_csv(huawei_test_pred_file)
    df = df[df.label == 1]
    items = df.values.tolist()
    url_to_prob = {}
    for item in items:
        url = 'https://github.com/' + item[5] + '/commit/' + item[0]
        url_to_prob[url] = item[4]

    original_df_file = 'huawei_pos_test_with_prob.csv'
    or_df = pd.read_csv(original_df_file)
    huawei_pred = []
    items = or_df.values.tolist()
    for item in items:
        url = item[1]
        huawei_pred.append(url_to_prob[url])

    or_df['Huawei_pred_probability'] = huawei_pred

    or_df.to_csv('Huawei_pred_compare.csv')

    print()


def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            original_line = line[1:]
            line = line[1:].strip()
            if line.startswith(('//', '/*', '/**', '*', '*/', '#')):
                continue
            code = code + original_line + '\n'

    return code


def do_something_2():

    result_df = pd.read_csv('Huawei_pred_compare.csv')

    df = pd.read_csv(dataset_name)

    df = df[(df.label == 1) & (df.partition == 'test')]
    items = df.values.tolist()
    url_to_added_token_counts = {}
    url_to_removed_token_counts = {}
    url_to_diff = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = 'https://github.com/' + repo + '/commit/' + commit_id
        diff = item[6]
        if url not in url_to_diff:
            url_to_diff[url] = ''

        url_to_diff[url] = url_to_diff[url] + '\n' + diff

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    finish = 0
    for url, diff in url_to_diff.items():
        added_code = get_code_version(diff, True)
        inputs = tokenizer(added_code, return_tensors="pt")
        token_count = inputs.data['input_ids'].shape[1]
        url_to_added_token_counts[url] = token_count

        removed_code = get_code_version(diff, False)
        inputs = tokenizer(removed_code, return_tensors="pt")
        token_count = inputs.data['input_ids'].shape[1]
        url_to_removed_token_counts[url] = token_count

        finish += 1
        print(finish)

    fail_count = 0
    huawei_fail_count = 0
    for item in result_df.values.tolist():
        url = item[2]
        pred = item[5]
        huawei_pred = item[6]
        added_token = url_to_added_token_counts[url]
        removed_token = url_to_removed_token_counts[url]
        if added_token > 512 or removed_token > 512:
            if pred > 0.5:
                fail_count += 1
            if huawei_pred > 0.5:
                huawei_fail_count += 1

    print("Long commit failed count: {}".format(fail_count))
    print("Huawei long commit failed count: {}".format(huawei_fail_count))


def read_url_to_token_count():
    url_to_added_count = {}
    url_to_removed_count = {}

    df = pd.read_csv('huawei_dataset_url_to_token_count.csv')

    for item in df.values.tolist():
        url_to_added_count['https://github.com/' + item[1]] = item[2]
        url_to_removed_count['https://github.com/' + item[1]] = item[3]

    return url_to_added_count, url_to_removed_count


def calculate_result(df):
    # url_to_added_count, url_to_removed_count = read_url_to_token_count()
    # items = df.values.tolist()
    # y_pred = []
    # y_true = []
    # y_prob = []

    # for item in items:
    #     url = item[5] + '/commit/' + item[0]
    #     added_count = url_to_added_count[url]
    #     removed_count = url_to_removed_count[url]
    #     if added_count > 512 or removed_count > 512:
    #         y_pred.append(item[3])
    #         y_true.append(item[1])
    #         y_prob.append(item[4])

    y_pred = df['pred_label']
    y_true = df['label']
    y_prob = df['pred_label_prob']
    print("Precision: {}".format(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
    print("Recall: {}".format(metrics.recall_score(y_pred=y_pred, y_true=y_true)))
    print("F1: {}".format(metrics.f1_score(y_pred=y_pred, y_true=y_true)))
    print("ROC_AUC: {}".format(metrics.roc_auc_score(y_score=y_prob, y_true=y_true)))
    print("PR_AUC: {}".format(metrics.average_precision_score(y_score=y_prob, y_true=y_true)))


def read_result():
    prob_file = 'huawei_comparison_classifier_prob_python.txt'
    df = pd.read_csv(prob_file, header=None)
    test_index_to_prob = {tup[0]: tup[1] for tup in df.values.tolist()}
    lang = 'python'

    patch_train, patch_test, patch_validation, label_train, label_test, label_validation, test_index_to_loc_mod, test_index_to_label, test_index_to_url = huawei_pure_classifier.get_data(
        lang=lang)
    y_prob = []
    y_test = []
    for index, prob in test_index_to_prob.items():
        y_prob.append(prob)
        y_test.append(test_index_to_label[index])

    print("PR_AUC: {}".format(metrics.average_precision_score(y_score=y_prob, y_true=y_test)))


def get_high_false_positives():
    result_path = 'huawei_comparison_slicing_classifier_prob_java.txt'
    result_df = pd.read_csv(result_path, header=None)
    index_to_prob = {tup[0]: tup[1] for tup in result_df.values.tolist()}
    patch_train, patch_test, patch_validation, label_train, label_test, label_validation, test_index_to_log_mod, test_index_to_label, test_index_to_url \
        = huawei_pure_classifier.get_data(lang='java')

    items = []
    for index, prob in index_to_prob.items():
        label = test_index_to_label[index]
        url = 'https://github.com/' + test_index_to_url[index]
        if label == 0 and prob > 0.7:
            items.append((url, '{0:.3f}'.format(prob)))

    df = pd.DataFrame(items, columns=['url', 'pred_prob'])

    df.to_csv("highly_false_positives.csv")


def get_sub_dataset_():
    df = pd.read_csv('ase_dataset_sept_19_2021.csv')
    df_pos = df[df.label == 1]
    train_len = len(df_pos[df_pos.partition == 'train'])
    test_len = len(df_pos[df_pos.partition == 'test'])
    val_len = len(df_pos[df_pos.partition == 'val'])
    df_neg_train = df[(df.label == 0) & (df.partition == 'train')].sample(n=5 * train_len)
    df_neg_test = df[(df.label == 0) & (df.partition == 'test')].sample(n=5 * test_len)
    df_neg_val = df[(df.label == 0) & (df.partition == 'val')].sample(n=5 * val_len)
    new_df = pd.concat([df_pos, df_neg_train, df_neg_test, df_neg_val])

    new_df.to_csv('huawei_sub_dataset_1_5.csv')


def do_something_3():
    df = pd.read_csv('ase_dataset_sept_19_2021.csv')
    df_test_pos = df[(df.label == 1) & (df.partition == 'test')][['commit_id', 'repo', 'diff', 'label', 'PL']]
    url_to_diff = {}
    url_to_pl = {}
    for item in df_test_pos.values.tolist():
        url = 'https://github.com/' + item[1] + '/commit/' + item[0]
        diff = item[2]
        PL = item[4]
        if url not in url_to_diff:
            url_to_diff[url] = ''

        url_to_diff[url] = url_to_diff[url] + '\n' + diff
        url_to_pl[url] = PL

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    url_to_token_count = {}
    for url, diff in url_to_diff.items():
        added_code = get_code_version(diff, True)
        removed_code = get_code_version(diff, False)

        inputs = tokenizer(added_code, return_tensors="pt")
        added_token_count = inputs.data['input_ids'].shape[1]

        inputs = tokenizer(removed_code, return_tensors="pt")
        removed_token_count = inputs.data['input_ids'].shape[1]

        url_to_token_count[url] = max(added_token_count, removed_token_count)

    result_file = 'huawei_pos_test_with_prob.csv'

    result_df = pd.read_csv(result_file)
    result_df = result_df[['commit_url', 'Notes', 'Can program slicing help','predicted_probability_for_VF']]
    url_list = result_df['commit_url'].values.tolist()
    token_count_list = []
    pl_list = []
    for url in url_list:
        token_count_list.append(url_to_token_count[url])
        pl_list.append(url_to_pl[url])

    result_df['token_count'] = token_count_list
    result_df['PL'] = pl_list

    result_df.to_csv('huawei_pos_test_21102021.csv')

    # df.loc[(df.a == 10) & (df.b == 20), ['x', 'y']].plot(title='a: 10, b: 20')


def get_relation(df, token_count, low_prob, high_prob, lang):
    sub_df = df[(df.token_count > token_count) & (low_prob <= df.pred) & (df.pred <= high_prob)]
    print("{}, Total data {}, filter by token count {}, low_prob {}, high_prob {}, found {}, rate {}"
          .format(lang, len(df), token_count, low_prob, high_prob, len(sub_df), len(sub_df)/len(df)))

def compare_result():
    result_file = 'huawei_pos_test_21102021.csv'
    df = pd.read_csv(result_file)

    df_java = df[df.PL == 'java'][['pred', 'token_count']]
    df_python = df[df.PL == 'python'][['pred', 'token_count']]

    df_java = df_java[(df.token_count > 512)]
    df_python = df_python[(df.token_count > 512)]

    true_java = df_java[(df.pred >= 0.5)]
    false_java = df_java[(df.pred < 0.5)]

    true_python = df_python[(df.pred >= 0.5)]
    false_python = df_python[(df.pred < 0.5)]

    print()
    # print(len(sub_java))
    # print(len(sub_python))
    #
    # get_relation(df_java, 600, 0, 0.5, 'java')
    # get_relation(df_python, 600, 0, 0.5, 'python')
    #
    # get_relation(df_java, 600, 0.1, 0.2, 'java')
    # get_relation(df_python, 600, 0.1, 0.2, 'python')
    #
    # get_relation(df_java, 600, 0.0, 0.5, 'java')
    # get_relation(df_python, 600, 0.0, 0.5, 'python')
    # print()
    # df_java.plot.scatter(title='Relation of predicted probability and token count in Java', x='pred', y ='token_count')
    # plt.show()
    #
    # df_python.plot.scatter(title='Relation of predicted probability and token count in python', x='pred', y ='token_count')
    # plt.show()


def extract_subdataset_with_slicing():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    print(len(df))
    items = df.values.tolist()

    url_to_added_loc = {}
    url_to_deleted_loc = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        added_loc = item[13]
        deleted_loc = item[14]
        if url not in url_to_added_loc:
            url_to_added_loc[url] = 0
        if url not in url_to_deleted_loc:
            url_to_deleted_loc[url] = 0

        url_to_added_loc[url] += added_loc
        url_to_deleted_loc[url] += deleted_loc

    url_to_index = {}
    with open('url_to_index.txt', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            url_to_index[row[0]] = row[1]

    accept_url_set = set()
    for url in url_to_added_loc.keys():
        added_loc = url_to_added_loc[url]
        deleted_loc = url_to_deleted_loc[url]

        if added_loc <= 10 and deleted_loc <= 10:
            accept_url_set.add(url)

    print()
        # if added_loc <= 15 and deleted_loc <= 15:
        #     previous_slicing_result_path = slicing_result_folder_path + '/' + url_to_index[url] + '_previous.txt'
        #     update_slicing_result_path = slicing_result_folder_path + '/' + url_to_index[url] + '_update.txt'
        #     if not os.path.isfile(previous_slicing_result_path) and not os.path.isfile(update_slicing_result_path):
        #         continue
        #     if os.path.isfile(previous_slicing_result_path):
        #         with open(previous_slicing_result_path, 'r') as reader:
        #             lines = reader.read().split('\n')
        #             if len(lines) - deleted_loc >= 3:
        #                 accept_url_set.add(url)
        #                 continue
        #
        #     if os.path.isfile(update_slicing_result_path):
        #         with open(update_slicing_result_path, 'r') as reader:
        #             lines = reader.read().split('\n')
        #             if len(lines) - added_loc >= 3:
        #                 accept_url_set.add(url)
        #                 continue


    # new_items = []
    # for item in items:
    #     commit_id = item[0]
    #     repo = item[1]
    #     url = repo + '/commit/' + commit_id
    #     if url in accept_url_set:
    #         new_items.append(item)
    #
    # # new_df = pd.DataFrame(new_items, columns=df.columns.values.tolist())
    # with open('huawei_csv_subset_slicing.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(df.columns.values.tolist())
    #     for item in new_items:
    #         writer.writerow(item)
    # # new_df.to_csv('huawei_sub_dataset_test_slicing_edited.csv')


def extract_context_from_diff(diff):
    contexts = []
    lines = diff.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('@@'):
            contexts.append(line.split('@@')[-1])

    return contexts

def extract_function_context():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[(df.partition == 'test') & (df.label == 1)]
    items = df.values.tolist()
    url_to_diff = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        diff = item[6]

        if url not in url_to_diff:
            url_to_diff[url] = ''

        url_to_diff[url] = url_to_diff[url] + diff + '\n'

    contexts_list = []
    for url, diff in url_to_diff.items():
        contexts = extract_context_from_diff(diff)
        contexts_list.append((url,contexts))
    print()


def url_to_token_length():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    items = df.values.tolist()
    url_to_diff = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        diff = item[6]

        if url not in url_to_diff:
            url_to_diff[url] = ''

        url_to_diff[url] = url_to_diff[url] + diff + '\n'

    total = len(url_to_diff)
    urls = []
    token_count_added_code = []
    token_count_removed_code = []

    count = 0
    for url, diff in url_to_diff.items():
        count += 1
        print("process: {}/{}".format(count, total))
        urls.append(url)
        added_code = get_code_version(diff, True)
        removed_code = get_code_version(diff, False)

        inputs = tokenizer(added_code, return_tensors="pt")
        token_count = inputs.data['input_ids'].shape[1]
        token_count_added_code.append(token_count)

        inputs = tokenizer(removed_code, return_tensors="pt")
        token_count = inputs.data['input_ids'].shape[1]
        token_count_removed_code.append(token_count)

    data = {'url': urls, 'added_token_count': token_count_added_code, 'removed_token_count': token_count_removed_code}

    df = pd.DataFrame(data)

    df.to_csv('huawei_dataset_url_to_token_count.csv')


def calculate_patch_classifier_result():
    result_path = 'huawei_comparison_slicing_classifier_prob_python.txt'
    result_df = pd.read_csv(result_path, header=None)
    index_to_prob = dict(result_df.values.tolist())
    url_to_prob = {}
    url_to_label = {}
    patch_train, patch_test, patch_validation, label_train, label_test, label_validation, test_index_to_loc_mod, test_index_to_label, test_index_to_url = huawei_pure_classifier.get_data(
        lang='python')

    for index, prob in index_to_prob.items():
        url_to_prob[test_index_to_url[index]] = prob
        url_to_label[test_index_to_url[index]] = test_index_to_label[index]

    url_to_added_count, url_to_removed_count = read_url_to_token_count()

    y_pred = []
    y_true = []
    y_prob = []
    commit_count = 0
    pos_count = 0
    neg_count = 0
    for url, prob in url_to_prob.items():
        added_count = url_to_added_count[url]
        removed_count = url_to_removed_count[url]
        if added_count > 512 or removed_count > 512:
            commit_count += 1
            if prob >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
            label = url_to_label[url]
            if label == 0:
                neg_count += 1
            else:
                pos_count += 1
            y_true.append(url_to_label[url])
            y_prob.append(prob)

    print("Total commit: {}".format(commit_count))
    print("Neg count: {}".format(neg_count))
    print("Pos count: {}".format(pos_count))
    print("Precision: {}".format(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
    print("Recall: {}".format(metrics.recall_score(y_pred=y_pred, y_true=y_true)))
    print("F1: {}".format(metrics.f1_score(y_pred=y_pred, y_true=y_true)))
    print("ROC_AUC: {}".format(metrics.roc_auc_score(y_score=y_prob, y_true=y_true)))
    print("PR_AUC: {}".format(metrics.average_precision_score(y_score=y_prob, y_true=y_true)))


def get_dataset_info():
    dataset_name = 'huawei_csv_subset_slicing_limited_10.csv'
    all_commits = pd.read_csv(dataset_name)

    py = all_commits[all_commits.PL == 'python']
    java = all_commits[all_commits.PL == 'java']

    # Java first: partition into train/val/test and check # of commits
    print("Java VF vs NVF for train/val/test")
    java_train = java[java.partition == "train"]
    java_val = java[java.partition == "val"]
    java_test = java[java.partition == "test"]
    print(java_train.drop_duplicates(subset='commit_id').label.value_counts())
    print(java_val.drop_duplicates(subset='commit_id').label.value_counts())
    print(java_test.drop_duplicates(subset='commit_id').label.value_counts())

    # Python: partition into train/val/test and check # of commits
    print("Py VF vs NVF for train/val/test")
    py_train = py[py.partition == "train"]
    py_val = py[py.partition == "val"]
    py_test = py[py.partition == "test"]
    print(py_train.drop_duplicates(subset='commit_id').label.value_counts())
    print(py_val.drop_duplicates(subset='commit_id').label.value_counts())
    print(py_test.drop_duplicates(subset='commit_id').label.value_counts())


def print_result_to_csv():
    java_result_path = 'huawei_pure_classifier_prob_01112021java.txt'
    python_result_path = 'huawei_pure_classifier_prob_01112021python.txt'

    _, _, _, _, _, _, _, java_test_index_to_label, java_test_index_to_url = huawei_pure_classifier.get_data('java')
    df_result_java = pd.read_csv(java_result_path)
    df_result_java = df_result_java[df_result_java.y_test == 1]
    java_url = []
    java_pl = []
    for index in df_result_java['index'].values.tolist():
        java_url.append(java_test_index_to_url[index])
        java_pl.append('java')
    df_result_java['url'] = java_url
    df_result_java['PL'] = java_pl

    _, _, _, _, _, _, _, python_test_index_to_label, python_test_index_to_url = huawei_pure_classifier.get_data('python')
    df_result_python = pd.read_csv(python_result_path)
    df_result_python = df_result_python[df_result_python.y_test == 1]
    python_url = []
    python_pl = []
    for index in df_result_python['index'].values.tolist():
        python_url.append(python_test_index_to_url[index])
        python_pl.append('python')
    df_result_python['url'] = python_url
    df_result_python['PL'] = python_pl

    df_result = pd.concat([df_result_java, df_result_python])


    huawei_test_pred_file = 'test_dataset_predictions.csv'
    huawei_df = pd.read_csv(huawei_test_pred_file)
    huawei_df = huawei_df[huawei_df.label == 1]
    items = huawei_df.values.tolist()
    url_to_prob = {}
    for item in items:
        url = 'https://github.com/' + item[5] + '/commit/' + item[0]
        url_to_prob[url] = item[4]

    huawei_pred = []
    for url in df_result['url'].values.tolist():
        huawei_pred.append(url_to_prob[url])

    df_result['huawei_pred'] = huawei_pred

    df_result = df_result[['url', 'PL', 'pred_prob', 'huawei_pred']]

    df_result = df_result.sort_values(by=['pred_prob'], ascending=False)

    df_result.to_csv('result_prob_sorted.csv', index=False)


def get_url_to_no_files():
    print("Retrieving url to files...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo']]

    url_to_files = {}
    for item in df.values.tolist():
        url = 'https://github.com/' + item[1] + '/commit/' + item[0]
        if url not in url_to_files:
            url_to_files[url] = 0

        url_to_files[url] = url_to_files[url] + 1

    return url_to_files


def filter_row_by_no_files(row):
    url = 'https://github.com/' + row['repo'] + '/commit/' + row['commit_id']
    if url_to_files[url] > 2 and url_to_added_count[url] <= 512 and url_to_removed_count[url] <= 512:
        return False
    else:
        return True


def read_huawei_result():
    print("Calculating Huawei result by number of files...")
    huawei_test_pred_file = 'test_dataset_predictions.csv'
    df = pd.read_csv(huawei_test_pred_file)

    df = df[df.apply(filter_row_by_no_files, axis=1)]
    print("Result on Java project...")
    calculate_result(df[df.PL == 'java'])

    print("Result on Python project...")
    calculate_result(df[df.PL == 'python'])


def read_patch_classifier_result_by_file():
    java_result_path = 'huawei_pure_classifier_prob_01112021java.txt'
    python_result_path = 'huawei_pure_classifier_prob_01112021python.txt'

    df_result = pd.read_csv(python_result_path)
    _, _, _, _, _, _, _, java_test_index_to_label, java_test_index_to_url = huawei_pure_classifier.get_data('python')
    y_pred = []
    y_true = []
    y_prob = []
    for item in df_result.values.tolist():
        index = item[0]
        url = java_test_index_to_url[index]
        ok = url_to_files[url] > 2 and url_to_added_count[url] <= 512 and url_to_removed_count[url] <= 512
        if not ok:
            y_prob.append(item[1])
            y_pred.append(item[2])
            y_true.append(item[3])

    print("Precision: {}".format(metrics.precision_score(y_pred=y_pred, y_true=y_true)))
    print("Recall: {}".format(metrics.recall_score(y_pred=y_pred, y_true=y_true)))
    print("F1: {}".format(metrics.f1_score(y_pred=y_pred, y_true=y_true)))
    print("ROC_AUC: {}".format(metrics.roc_auc_score(y_score=y_prob, y_true=y_true)))
    print("PR_AUC: {}".format(metrics.average_precision_score(y_score=y_prob, y_true=y_true)))



def write_url_to_hunk_count():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename']]
    items = df.to_numpy().tolist()

    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}
    url_to_hunk_count = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        diff = item[3]
        label = item[4]
        pl = item[5]

        if url not in url_to_hunk_count:
            url_to_hunk_count[url] = 0

        url_to_hunk_count[url] += len(preprocess_variant_3.get_hunk_from_diff(diff))
        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    with open('huawei_dataset_url_to_hunk_count.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['url', 'hunk_count', 'label', 'partition', 'pl'])
        for key in url_to_hunk_count.keys():
            url = key
            hunk_count = url_to_hunk_count[key]
            label = url_to_label[key]
            partition = url_to_partition[key]
            pl = url_to_pl[key]
            writer.writerow([url, hunk_count, label, partition, pl])


if __name__ == '__main__':
    # url_to_files = get_url_to_no_files()
    # url_to_added_count, url_to_removed_count = read_url_to_token_count()
    # read_patch_classifier_result_by_file()
    write_url_to_hunk_count()