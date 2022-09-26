import pandas as pd
from tqdm import tqdm
import os.path
import json
import re
import math


def get_data_from_saved_file(file_info_name, need_pl=False):
    with open(file_info_name, 'r') as reader:
        data = json.loads(reader.read())

    if need_pl:
        return data['url_data'], data['label_data'], data['url_to_pl'], data['url_to_label']
    else:
        return data['url_data'], data['label_data']


def get_data(dataset_name, need_pl=False):
    file_info_name = 'info_' + dataset_name + '.json'
    if os.path.isfile(file_info_name):
        return get_data_from_saved_file(file_info_name, need_pl)

    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'PL', 'label']]
    items = df.to_numpy().tolist()

    url_train, url_val, url_val_java, url_val_python, url_test_java, url_test_python = [], [], [], [], [], []
    label_train, label_val, label_val_java, label_val_python, label_test_java, label_test_python = [], [], [], [], [], []
    url_to_pl = {}
    url_to_label = {}
    for item in tqdm(items):
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        pl = item[3]
        label = item[4]
        url_to_pl[url] = pl
        url_to_label[url] = label
        if partition == 'train':
            if url not in url_train:
                url_train.append(url)
                label_train.append(label)
        elif partition == 'val':
            if url not in url_val:
                url_val.append(url)
                label_val.append(label)
            if pl == 'java' and url not in url_val_java:
                url_val_java.append(url)
                label_val_java.append(label)
            if pl == 'python' and url not in url_val_python:
                url_val_python.append(url)
                label_val_python.append(label)

        elif partition == 'test':
            if pl == 'java' and url not in url_test_java:
                url_test_java.append(url)
                label_test_java.append(label)
            elif pl == 'python' and url not in url_test_python:
                url_test_python.append(url)
                label_test_python.append(label)
        else:
            Exception("Invalid partition: {}".format(partition))

    print("Finish reading dataset")
    url_data = {'train': url_train, 'val': url_val, 'val_java': url_val_java, 'val_python': url_val_python,
                'test_java': url_test_java, 'test_python': url_test_python}
    label_data = {'train': label_train, 'val': label_val, 'val_java': label_val_java, 'val_python': label_val_python,
                'test_java': label_test_java, 'test_python': label_test_python}

    data = {'url_data': url_data, 'label_data': label_data, 'url_to_pl': url_to_pl, 'url_to_label' : url_to_label}

    json.dump(data, open(file_info_name, 'w'))

    if need_pl:
        return url_data, label_data, url_to_pl, url_to_label
    else:
        return url_data, label_data



def extract_security_dataset(dataset_name, output_path):
    java_sec_url_set, python_sec_url_set = filter_security_changes_by_keywords(dataset_name)
    print(len(java_sec_url_set))
    print(len(python_sec_url_set))
    print("Reading dataset....")
    df = pd.read_csv(dataset_name)

    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename', 'msg']]
    df = df[df.partition == 'test']
    items = df.to_numpy().tolist()
    sec_items = []

    for item in items:
        label = item[4]
        url = item[1] + '/commit/' + item[0]

        if label == 1 or url in java_sec_url_set or url in python_sec_url_set:
            sec_items.append(item)

    
    sec_df = pd.DataFrame(sec_items, columns= ['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename', 'msg']) 

    sec_df.to_csv(output_path, encoding='utf-8')


def filter_security_changes_by_keywords(dataset_name):
    print("Reading dataset....")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'PL', 'label', 'msg']]

    df = df[df.label == 0]
    df = df[df.partition == 'test']

    items = df.to_numpy().tolist()  

    python_sec_url_set = set()
    java_sec_url_set = set()

    sec_message_set = set()
    strong_regex = re.compile(r'(?i)(denial.of.service|remote.code.execution|\bopen.redirect|OSVDB|\bXSS\b|\bReDoS\b|\bNVD\b|malicious|x−frame−options|attack|cross.site|exploit|directory.traversal|\bRCE\b|\bdos\b|\bXSRF\b|clickjack|session.fixation|hijack|advisory|insecure|security|\bcross−origin\b|unauthori[z|s]ed|infinite.loop)')
    medium_regex =re.compile(r'(?i)(authenticat(e|ion)|bruteforce|bypass|constant.time|crack|credential|\bDoS\b|expos(e|ing)|hack|harden|injection|lockout|overflow|password|\bPoC\b|proof.of.concept|poison|privelage|\b(in)?secur(e|ity)|(de)?serializ|spoof|timing|traversal)')

    for item in tqdm(items):
        message = item[5]
        url = item[1] + '/commit/' + item[0]
        pl = item[3]

        if not isinstance(message, str) and math.isnan(message):
            continue
        
        m = strong_regex.search(message)
        n = medium_regex.search(message)
        if m or n:
            if pl == 'java':
                java_sec_url_set.add(url)
            else:
                python_sec_url_set.add(url)


            sec_message_set.add(message)

    return java_sec_url_set, python_sec_url_set



if __name__== '__main__':

    dataset_name = 'ase_dataset_sept_19_2021.csv'
    sec_dataset_name = 'ase_surity_sub_dataset.csv'

    extract_security_dataset(dataset_name, sec_dataset_name)
 
