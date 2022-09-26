import torch
from torch import nn as nn
import os
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from entities import VariantThreeFineTuneOnlyDataset
from model import VariantThreeFineTuneOnlyClassifier
from pytorchtools import EarlyStopping
import pandas as pd
from tqdm import tqdm
import utils
from transformers import RobertaTokenizer
import preprocess_variant_3
import preprocess_variant_1

# dataset_name = 'ase_dataset_sept_19_2021.csv'
dataset_name = 'huawei_sub_dataset.csv'
directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

BEST_MODEL_PATH = 'model/patch_variant_3_finetune_best_model.sav'
FINE_TUNED_MODEL_PATH = 'model/patch_variant_3_finetuned_model.sav'

FINETUNE_EPOCH = 1

NUMBER_OF_EPOCHS = 1
EARLY_STOPPING_ROUND = 5

TRAIN_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 256
HIDDEN_DIM = 768

NUMBER_OF_LABELS = 2


def train(model, learning_rate, number_of_epochs, training_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for index, (id_batch, url_batch, input_batch, mask_batch, label_batch) in enumerate(training_generator):
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)
            outs = model(input_batch, mask_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, commit {}/{} total loss {}, average loss {}"
                      .format(current_batch, (index + 1) * TRAIN_BATCH_SIZE, len(training_generator) * TRAIN_BATCH_SIZE, np.sum(train_losses), np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))

        torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)


        # if epoch + 1 == FINETUNE_EPOCH:
        #     torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
        #     if not isinstance(model, nn.DataParallel):
        #         model.freeze_codebert()
        #     else:
        #         model.module.freeze_codebert()

    return model


def get_data():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        diff = item[3]
        label = item[4]
        pl = item[5]

        if url not in url_to_diff:
            url_to_diff[url] = []

        url_to_diff[url].extend(preprocess_variant_3.get_hunk_from_diff(diff))
        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    patch_train, patch_val, patch_test_java, patch_test_python = [], [], [], []
    label_train, label_val, label_test_java, label_test_python = [], [], [], []
    url_train, url_val, url_test_java, url_test_python = [], [], [], []

    for key in url_to_diff.keys():
        url = key
        diff = url_to_diff[key]
        label = url_to_label[key]
        partition = url_to_partition[key]
        pl = url_to_pl[key]
        if partition == 'train':
            patch_train.append(diff)
            label_train.append(label)
            url_train.append(url)
        elif partition == 'test':
            if pl == 'java':
                patch_test_java.append(diff)
                label_test_java.append(label)
                url_test_java.append(url)
            elif pl == 'python':
                patch_test_python.append(diff)
                label_test_python.append(label)
                url_test_python.append(url)
            else:
                raise Exception("Invalid programming language: {}".format(partition))
        elif partition == 'val':
            patch_val.append(diff)
            label_val.append(label)
            url_val.append(url)
        else:
            raise Exception("Invalid partition: {}".format(partition))

    print("Finish reading dataset")
    patch_data = {'train': patch_train, 'val': patch_val,
                  'test_java': patch_test_java, 'test_python': patch_test_python}

    label_data = {'train': label_train, 'val': label_val,
                  'test_java': label_test_java, 'test_python': label_test_python}

    url_data = {'train': url_train, 'val': url_val,
                'test_java': url_test_java, 'test_python': url_test_python}

    return patch_data, label_data, url_data


def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'], inputs.data['attention_mask']


def retrieve_patch_data(all_data, all_label, all_url):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Preparing tokenizer data...")

    id_to_label = {}
    id_to_url = {}
    id_to_input = {}
    id_to_mask = {}
    index = 0
    for i, hunk_list in tqdm(enumerate(all_data)):
        code_list = []

        for count, file in enumerate(hunk_list):
            added_code = preprocess_variant_3.get_code_version(diff=file, added_version=True)
            deleted_code = preprocess_variant_3.get_code_version(diff=file, added_version=False)

            code = added_code + tokenizer.sep_token + deleted_code
            code_list.append(code)

        input_ids_list, mask_list = get_input_and_mask(tokenizer, code_list)
        for j in range(len(input_ids_list)):
            id_to_input[index] = input_ids_list[j]
            id_to_mask[index] = mask_list[j]
            id_to_label[index] = all_label[i]
            id_to_url[index] = all_url[i]
            index += 1

    return id_to_input, id_to_mask, id_to_label, id_to_url


def do_train():
    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(BEST_MODEL_PATH))
    patch_data, label_data, url_data = get_data()

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []

    index = 0
    for i, hunk_list in enumerate((patch_data['train'])):
        for j in range(len(hunk_list)):
            train_ids.append(index)
            index += 1

    all_data = patch_data['train']
    all_label = label_data['train']
    all_url = url_data['train']

    print("Preparing commit patch data...")
    id_to_input, id_to_mask, id_to_label, id_to_url = retrieve_patch_data(all_data, all_label, all_url)
    print("Finish preparing commit patch data")

    training_set = VariantThreeFineTuneOnlyDataset(train_ids, id_to_label, id_to_url, id_to_input, id_to_mask)
    training_generator = DataLoader(training_set, **TRAIN_PARAMS)

    model = VariantThreeFineTuneOnlyClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator)


if __name__ == '__main__':
    do_train()
