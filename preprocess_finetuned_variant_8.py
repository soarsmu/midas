from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import json
import os
import torch
import tqdm
from torch import cuda
from torch import nn as nn
from model import VariantEightFineTuneOnlyClassifier
import matplotlib.pyplot as plt


EMBEDDING_DIRECTORY = '../finetuned_embeddings/variant_8'
FINE_TUNED_MODEL_PATH = 'model/patch_variant_8_finetuned_model.sav'

directory = os.path.dirname(os.path.abspath(__file__))

dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset_new.csv'

CODE_LINE_LENGTH = 64

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_input_and_mask(tokenizer, code_list):
    inputs = tokenizer(code_list, padding=True, max_length=CODE_LINE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'], inputs.data['attention_mask']


def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '/*', '*', '*/', '#')) or line.strip() == '':
                continue
            code = code + line + '\n'

    return code


def get_embeddings(code_list, start, length, tokenizer, codebert):
    input_ids, attention_mask = get_input_and_mask(tokenizer, code_list[start: start + length])

    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        embeddings = codebert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
    embeddings = embeddings.tolist()

    return embeddings


def get_line_embeddings(code_list, tokenizer, code_bert):
    if len(code_list) == 0:
        return []

    embeddings = []
    index = 0
    length_limit = 100
    while index + length_limit < len(code_list):
        embeddings.extend(get_embeddings(code_list, index, length_limit, tokenizer, code_bert))
        index = index + length_limit

    embeddings.extend(get_embeddings(code_list, index, len(code_list) - index, tokenizer, code_bert))
    # process all lines in one

    return embeddings


def line_empty(line):
    if line.strip() == '':
        return True
    else:
        return False


def get_line_from_code(sep_token, code):
    lines = []
    for line in code.split('\n'):
        if not line_empty(line):
            lines.append(sep_token + line)

    return lines


def write_embeddings_to_files(removed_embeddings, added_embeddings, removed_url_list, added_url_list):
    url_set = set()
    url_to_removed_embeddings = {}
    for index, url in enumerate(removed_url_list):
        if url not in url_to_removed_embeddings:
            url_set.add(url)
            url_to_removed_embeddings[url] = []
        url_to_removed_embeddings[url].append(removed_embeddings[index])

    url_to_added_embeddings = {}
    for index, url in enumerate(added_url_list):
        if url not in url_to_added_embeddings:
            url_set.add(url)
            url_to_added_embeddings[url] = []
        url_to_added_embeddings[url].append(added_embeddings[index])

    url_to_data = {}
    for url in url_set:
        before_data = []
        after_data = []
        if url in url_to_removed_embeddings:
            before_data = url_to_removed_embeddings[url]
        if url in url_to_added_embeddings:
            after_data = url_to_added_embeddings[url]

        data = {'before': before_data, 'after': after_data}
        url_to_data[url] = data
    for url, data in url_to_data.items():
        file_path = os.path.join(directory, EMBEDDING_DIRECTORY + '/' + url.replace('/', '_') + '.txt')
        json.dump(data, open(file_path, 'w'))


def get_data():
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = VariantEightFineTuneOnlyClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(FINE_TUNED_MODEL_PATH))
    code_bert = model.module.code_bert

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        code_bert = nn.DataParallel(code_bert)
    code_bert = code_bert.to(device)

    code_bert.eval()

    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename']]
    items = df.to_numpy().tolist()

    url_to_diff = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        diff = item[3]

        if url not in url_to_diff:
            url_to_diff[url] = ''

        url_to_diff[url] = url_to_diff[url] + diff + '\n'

    removed_code_list = []
    added_code_list = []
    removed_url_list = []
    added_url_list = []
    for url, diff in tqdm.tqdm(url_to_diff.items()):
        file_path = os.path.join(directory, EMBEDDING_DIRECTORY + '/' + url.replace('/', '_') + '.txt')
        if os.path.isfile(file_path):
            continue

        removed_code = get_code_version(diff, False)
        added_code = get_code_version(diff, True)

        new_removed_code_list = get_line_from_code(tokenizer.sep_token, removed_code)
        new_added_code_list = get_line_from_code(tokenizer.sep_token, added_code)

        for i in range(len(new_removed_code_list)):
            removed_url_list.append(url)

        for i in range(len(new_added_code_list)):
            added_url_list.append(url)

        removed_code_list.extend(new_removed_code_list)
        added_code_list.extend(new_added_code_list)

        if len(removed_code_list) >= 500 or len(added_code_list) >= 500:
            removed_embeddings = get_line_embeddings(removed_code_list, tokenizer, code_bert)
            added_embeddings = get_line_embeddings(added_code_list, tokenizer, code_bert)

            write_embeddings_to_files(removed_embeddings, added_embeddings, removed_url_list, added_url_list)

            removed_code_list = []
            added_code_list = []
            removed_url_list = []
            added_url_list = []

    if len(removed_code_list) > 0 or len(added_code_list) > 0:
        removed_embeddings = get_line_embeddings(removed_code_list, tokenizer, code_bert)
        added_embeddings = get_line_embeddings(added_code_list, tokenizer, code_bert)
        write_embeddings_to_files(removed_embeddings, added_embeddings, removed_url_list, added_url_list)


if __name__ == '__main__':
    get_data()
