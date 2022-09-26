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
from entities import VariantTwoFineTuneDataset
from model import VariantTwoFineTuneClassifier
from pytorchtools import EarlyStopping
from tqdm import tqdm
import pandas as pd
import preprocess_variant_1


dataset_name = 'huawei_sub_dataset.csv'
# dataset_name = 'ase_dataset_sept_19_2021.csv'

BEST_MODEL_PATH = 'model/patch_variant_2_finetune_best_model.sav'
FINE_TUNED_MODEL_PATH = 'model/patch_variant_2_finetuned_model.sav'

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')

FINETUNE_EPOCH = 1

LIMIT_FILE_COUNT = 5

NUMBER_OF_EPOCHS = 1
TRAIN_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EARLY_STOPPING_ROUND = 5

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CODE_LENGTH = 256
HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.3
NUMBER_OF_LABELS = 2


def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'], inputs.data['attention_mask']


def predict_test_data(model, testing_generator, device, need_prob=False):
    print("Testing...")
    y_pred = []
    y_test = []
    urls = []
    probs = []
    model.eval()
    with torch.no_grad():
        for id_batch, url_batch, input_list_batch, mask_list_batch, label_batch in tqdm(testing_generator):
            input_list_batch, mask_list_batch, label_batch \
                = input_list_batch.to(device), mask_list_batch.to(device), label_batch.to(device)

            outs = model(input_list_batch, mask_list_batch)
            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)

        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    if not need_prob:
        return precision, recall, f1, auc
    else:
        return precision, recall, f1, auc, urls, probs


def get_avg_validation_loss(model, validation_generator, loss_function):
    validation_loss = 0
    with torch.no_grad():
        for id_batch, url_batch, input_list_batch, mask_list_batch, label_batch in validation_generator:
            input_list_batch, mask_list_batch, label_batch \
                = input_list_batch.to(device), mask_list_batch.to(device), label_batch.to(device)
            outs = model(input_list_batch, mask_list_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            validation_loss += loss

    avg_val_los = validation_loss / len(validation_generator)

    return avg_val_los


def train(model, learning_rate, number_of_epochs, training_generator, val_generator,
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
    train_losses = []

    early_stopping = EarlyStopping(patience=EARLY_STOPPING_ROUND,
                                   verbose=True, path=BEST_MODEL_PATH)

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for id_batch, url_batch, input_list_batch, mask_list_batch, label_batch in training_generator:
            input_list_batch, mask_list_batch, label_batch \
                = input_list_batch.to(device), mask_list_batch.to(device), label_batch.to(device)
            outs = model(input_list_batch, mask_list_batch)
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
        train_losses = []

        model.eval()

        # print("Calculating validation loss...")
        # val_loss = get_avg_validation_loss(model, val_generator, loss_function)
        # print("Average validation loss of this iteration: {}".format(val_loss))
        # print("-" * 32)
        #
        # early_stopping(val_loss, model)
        #
        # print("Result on Java testing dataset...")
        # precision, recall, f1, auc = predict_test_data(model=model,
        #                                                testing_generator=test_java_generator,
        #                                                device=device)
        #
        # print("Precision: {}".format(precision))
        # print("Recall: {}".format(recall))
        # print("F1: {}".format(f1))
        # print("AUC: {}".format(auc))
        # print("-" * 32)
        #
        # print("Result on Python testing dataset...")
        # precision, recall, f1, auc = predict_test_data(model=model,
        #                                                testing_generator=test_python_generator,
        #                                                device=device)
        #
        # print("Precision: {}".format(precision))
        # print("Recall: {}".format(recall))
        # print("F1: {}".format(f1))
        # print("AUC: {}".format(auc))
        # print("-" * 32)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch + 1 == FINETUNE_EPOCH:
            torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
            if not isinstance(model, nn.DataParallel):
                model.freeze_codebert()
            else:
                model.module.freeze_codebert()
    return model


def retrieve_patch_data(all_data, all_label, all_url):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Preparing tokenizer data...")

    id_to_label = {}
    id_to_url = {}
    id_to_input_list = {}
    id_to_mask_list = {}
    for i, file_list in tqdm(enumerate(all_data)):
        code_list = []

        for count, file in enumerate(file_list):
            if count >= LIMIT_FILE_COUNT:
                continue

            added_code = preprocess_variant_1.get_code_version(diff=file, added_version=True)
            deleted_code = preprocess_variant_1.get_code_version(diff=file, added_version=False)

            code = added_code + tokenizer.sep_token + deleted_code
            code_list.append(code)

        # padding
        while len(code_list) < 5:
            code_list.append(tokenizer.sep_token)

        input_ids_list, mask_list = get_input_and_mask(tokenizer, code_list)
        id_to_input_list[i] = input_ids_list
        id_to_mask_list[i] = mask_list
        id_to_label[i] = all_label[i]
        id_to_url[i] = all_url[i]

    return id_to_input_list, id_to_mask_list, id_to_label, id_to_url


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
        url = repo + '/commit/' + commit_id
        partition = item[2]
        diff = item[3]
        label = item[4]
        pl = item[5]

        if url not in url_to_diff:
            url_to_diff[url] = []

        url_to_diff[url].append(diff)
        url_to_partition[url] = partition
        url_to_label[url] = label
        url_to_pl[url] = pl

    patch_train, patch_val, patch_test_java, patch_test_python = [], [], [], []
    label_train, label_val, label_test_java, label_test_python = [], [], [], []
    url_train, url_val, url_test_java, url_test_python = [], [], [], []

    print(len(url_to_diff.keys()))
    # diff here is diff list
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


def do_train():
    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(BEST_MODEL_PATH))
    patch_data, label_data, url_data = get_data()

    train_ids, val_ids, test_java_ids, test_python_ids = [], [], [], []

    index = 0
    for i in range(len(patch_data['train'])):
        train_ids.append(index)
        index += 1

    for i in range(len(patch_data['val'])):
        val_ids.append(index)
        index += 1

    for i in range(len(patch_data['test_java'])):
        test_java_ids.append(index)
        index += 1

    for i in range(len(patch_data['test_python'])):
        test_python_ids.append(index)
        index += 1

    all_data = patch_data['train'] + patch_data['val'] + patch_data['test_java'] + patch_data['test_python']
    all_label = label_data['train'] + label_data['val'] + label_data['test_java'] + label_data['test_python']
    all_url = url_data['train'] + url_data['val'] + url_data['test_java'] + url_data['test_python']

    print("Preparing commit patch data...")
    id_to_input_list, id_to_mask_list, id_to_label, id_to_url = retrieve_patch_data(all_data, all_label, all_url)
    print("Finish preparing commit patch data")

    training_set = VariantTwoFineTuneDataset(train_ids, id_to_label, id_to_url, id_to_input_list, id_to_mask_list)
    val_set = VariantTwoFineTuneDataset(val_ids, id_to_label, id_to_url, id_to_input_list, id_to_mask_list)
    test_java_set = VariantTwoFineTuneDataset(test_java_ids, id_to_label, id_to_url, id_to_input_list, id_to_mask_list)
    test_python_set = VariantTwoFineTuneDataset(test_python_ids, id_to_label, id_to_url, id_to_input_list, id_to_mask_list)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    val_generator = DataLoader(val_set, **VALIDATION_PARAMS)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    model = VariantTwoFineTuneClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          val_generator=val_generator,
          test_java_generator=test_java_generator,
          test_python_generator=test_python_generator)


if __name__ == '__main__':
    do_train()
