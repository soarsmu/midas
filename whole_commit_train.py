from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_scheduler

import pandas as pd

# dataset_name = 'huawei_csv_subset_slicing_limited_10.csv'
# dataset_name = 'huawei_sub_dataset.csv'
dataset_name = 'ase_dataset_sept_19_2021.csv'

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')

NUMBER_OF_EPOCHS = 15

TRAIN_BATCH_SIZE = 16

VALIDATION_BATCH_SIZE = 128

TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# random_seed = 109
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 512
HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.3
NUMBER_OF_LABELS = 2

model_path_prefix = model_folder_path + '/patch_classifier_variant_1_08112021_model_'


class PatchDataset(Dataset):
    def __init__(self, list_IDs, labels, id_to_input, id_to_mask):
        self.list_IDs = list_IDs
        self.labels = labels
        self.id_to_input = id_to_input
        self.id_to_mask = id_to_mask

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        id = self.list_IDs[index]
        input_id = self.id_to_input[id]
        mask = self.id_to_mask[id]
        y = self.labels[id]

        return int(id), input_id, mask, y


class PatchClassifier(nn.Module):
    def __init__(self):
        super(PatchClassifier, self).__init__()
        self.code_bert = RobertaModel.from_pretrained("microsoft/codebert-base", num_labels=2)

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(HIDDEN_DIM_DROPOUT_PROB)
        self.out_proj = nn.Linear(HIDDEN_DIM, NUMBER_OF_LABELS)

    def forward(self, input_batch, mask_batch):
        embeddings = self.code_bert(input_ids=input_batch, attention_mask=mask_batch)
        embeddings = embeddings.last_hidden_state[:, 0, :]

        x = self.relu(embeddings)
        x = self.drop_out(x)
        x = self.out_proj(x)

        return x, embeddings


def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'][0], inputs.data['attention_mask'][0]


def predict_test_data(model, testing_generator, device, need_prob_and_id=False):
    print("Testing...")
    y_pred = []
    y_test = []
    ids = []
    probs = []
    model.eval()
    with torch.no_grad():
        for id_batch, input_batch, mask_batch, label_batch in testing_generator:
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)

            outs = model(input_batch, mask_batch)[0]
            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            ids.extend(id_batch.tolist())
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    if not need_prob_and_id:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, ids, probs, y_pred, y_test


def train(model, learning_rate, number_of_epochs, training_generator, val_java_generator, val_python_generator,
          test_java_generator, test_python_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses, valid_losses = [], []
    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for id_batch, input_batch, mask_batch, label_batch in training_generator:
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)
            outs = model(input_batch, mask_batch)[0]
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
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))

        model.eval()

        print("Result on Java validation dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=val_java_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        print("Result on Python validation dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=val_python_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), model_path_prefix + '_patch_classifier_epoc_' + str(epoch) + '.sav')
        else:
            torch.save(model.state_dict(), model_path_prefix + '_patch_classifier.sav')

        print("Result on Java testing dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=test_java_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        print("Result on Python testing dataset...")
        precision, recall, f1, auc = predict_test_data(model=model,
                                                       testing_generator=test_python_generator,
                                                       device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

    return model


def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '*', '*/', '#')):
                continue
            code = code + line + '\n'

    return code


def retrieve_patch_data(all_data, all_label):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Preparing tokenizer data...")

    count = 0

    id_to_label = {}
    id_to_input = {}
    id_to_mask = {}
    for i in range(len(all_data)):
        added_code = get_code_version(diff=all_data[i], added_version=True)
        deleted_code = get_code_version(diff=all_data[i], added_version=False)

        # TODO: need to balance code between added_code and deleted_code due to data truncation?
        code = added_code + tokenizer.sep_token + deleted_code
        input_ids, mask = get_input_and_mask(tokenizer, code)
        id_to_input[i] = input_ids
        id_to_mask[i] = mask
        id_to_label[i] = all_label[i]
        count += 1
        if count % 1000 == 0:
            print("Number of records tokenized: {}/{}".format(count, len(all_data)))

    return id_to_input, id_to_mask, id_to_label


def get_data():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL']]
    items = df.to_numpy().tolist()

    url_to_diff = {}
    url_to_partition = {}
    url_to_label = {}
    url_to_pl = {}
    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = 'https://github.com/' + repo + '/commit/' + commit_id
        partition = item[2]
        diff = item[3]
        label = item[4]
        pl = item[5]

        if url not in url_to_diff:
            url_to_diff[url] = ''

        url_to_diff[url] = url_to_diff[url] + '\n' + diff
        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    patch_train, patch_val_java, patch_val_python, patch_test_java, patch_test_python = [], [], [], [], []
    label_train, label_val_java, label_val_python, label_test_java, label_test_python = [], [], [], [], []

    print(len(url_to_diff.keys()))
    for key in url_to_diff.keys():
        diff = url_to_diff[key]
        label = url_to_label[key]
        partition = url_to_partition[key]
        pl = url_to_pl[key]
        if partition == 'train':
            patch_train.append(diff)
            label_train.append(label)
        elif partition == 'test':
            if pl == 'java':
                patch_test_java.append(diff)
                label_test_java.append(label)
            elif pl == 'python':
                patch_test_python.append(diff)
                label_test_python.append(label)
            else:
                raise Exception("Invalid programming language: {}".format(partition))
        elif partition == 'val':
            if pl == 'java':
                patch_val_java.append(diff)
                label_val_java.append(label)
            elif pl == 'python':
                patch_val_python.append(diff)
                label_val_python.append(label)
            else:
                raise Exception("Invalid programming language: {}".format(partition))
        else:
            raise Exception("Invalid partition: {}".format(partition))

    print("Finish reading dataset")
    patch_data = {'train': patch_train, 'val_java': patch_val_java, 'val_python': patch_val_python,
                  'test_java': patch_test_java, 'test_python': patch_test_python}

    label_data = {'train': label_train, 'val_java': label_val_java, 'val_python': label_val_python,
                  'test_java': label_test_java, 'test_python': label_test_python}

    return patch_data, label_data


def do_train():
    print("Dataset name: {}".format(dataset_name))
    patch_data, label_data = get_data()

    train_ids, val_java_ids, val_python_ids, test_java_ids, test_python_ids = [], [], [], [], []
    index = 0
    for i in range(len(patch_data['train'])):
        train_ids.append(index)
        index += 1

    for i in range(len(patch_data['val_java'])):
        val_java_ids.append(index)
        index += 1

    for i in range(len(patch_data['val_python'])):
        val_python_ids.append(index)
        index += 1

    for i in range(len(patch_data['test_java'])):
        test_java_ids.append(index)
        index += 1

    for i in range(len(patch_data['test_python'])):
        test_python_ids.append(index)
        index += 1

    all_data = patch_data['train'] + patch_data['val_java'] + patch_data['val_python'] + patch_data['test_java'] + \
               patch_data['test_python']
    all_label = label_data['train'] + label_data['val_java'] + label_data['val_python'] + label_data['test_java'] + \
                label_data['test_python']

    print("Preparing commit patch data...")
    id_to_input, id_to_mask, id_to_label \
        = retrieve_patch_data(all_data, all_label)
    print("Finish preparing commit patch data")

    training_set = PatchDataset(train_ids, id_to_label, id_to_input, id_to_mask)
    val_java_set = PatchDataset(val_java_ids, id_to_label, id_to_input, id_to_mask)
    val_python_set = PatchDataset(val_python_ids, id_to_label, id_to_input, id_to_mask)
    test_java_set = PatchDataset(test_java_ids, id_to_label, id_to_input, id_to_mask)
    test_python_set = PatchDataset(test_python_ids, id_to_label, id_to_input, id_to_mask)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    val_java_generator = DataLoader(val_java_set, **VALIDATION_PARAMS)
    val_python_generator = DataLoader(val_python_set, **VALIDATION_PARAMS)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    model = PatchClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          val_java_generator=val_java_generator,
          val_python_generator=val_python_generator,
          test_java_generator=test_java_generator,
          test_python_generator=test_python_generator)


if __name__ == '__main__':
    do_train()
