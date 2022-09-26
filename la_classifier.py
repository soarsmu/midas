import os
import pandas as pd
import csv
import utils
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from metrices_calculator import calculate_effort, calculate_normalized_effort
dataset_name = 'ase_dataset_sept_19_2021.csv'


def get_loc_add():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_ADD']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_label = {}
    url_to_pl = {}
    url_to_loc_add = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        label = item[4]
        pl = item[5]
        loc_add = item[6]

        if url not in url_to_diff:
            url_to_diff[url] = ''
            url_to_loc_add[url] = 0

        url_to_label[url] = label
        url_to_pl[url] = pl
        url_to_loc_add[url] += loc_add

    return url_to_label, url_to_loc_add



def do_train():
    print("Reading data...")
    url_data, label_data = utils.get_data(dataset_name)
    url_to_label, url_to_loc_add = get_loc_add()

    X_train = []
    y_train = []
    for url in url_data['train']:
        X_train.append([url_to_loc_add[url]])
        y_train.append(url_to_label[url])

    print("Training classifier...")
    clf = LogisticRegression(random_state=109).fit(X_train, y_train)


    # Testing on Java
    print("Testing on Java...")

    X_test_java = []
    y_test_java = []
    for url in url_data['test_java']:
        X_test_java.append([url_to_loc_add[url]])
        y_test_java.append(url_to_label[url])

    pred_probs = clf.predict_proba(X_test_java)
    y_pred_java = []
    for prob in pred_probs:
        y_pred_java.append(prob[1])

    auc = metrics.roc_auc_score(y_true=y_test_java, y_score=y_pred_java)
    print("AUC on java: {}".format(auc))
    la_java_prob_path = 'probs/la_prob_java.txt'
    with open(la_java_prob_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(url_data['test_java']):
            writer.writerow([url, pred_probs[i][1]])

    calculate_effort(la_java_prob_path, 'java')
    calculate_normalized_effort(la_java_prob_path, 'java')

    # Testing on Python

    X_test_python = []
    y_test_python = []
    for url in url_data['test_python']:
        X_test_python.append([url_to_loc_add[url]])
        y_test_python.append(url_to_label[url])

    pred_probs = clf.predict_proba(X_test_python)
    y_pred_python = []
    for prob in pred_probs:
        y_pred_python.append(prob[1])

    auc = metrics.roc_auc_score(y_true=y_test_python, y_score=y_pred_python)
    print("AUC on python: {}".format(auc))
    la_python_prob_path = 'probs/la_prob_python.txt'
    with open(la_python_prob_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(url_data['test_python']):
            writer.writerow([url, pred_probs[i][1]])

    calculate_effort(la_java_prob_path, 'python')
    calculate_normalized_effort(la_java_prob_path, 'python')


if __name__ == '__main__':
    do_train()