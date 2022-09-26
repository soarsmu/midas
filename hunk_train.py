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
from entities import HunkDataset
from hunk_model import PatchClassifierByHunk
import pandas as pd
from tqdm import tqdm
import utils

dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset_new.csv'
directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

NUMBER_OF_EPOCHS = 100

TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

LEARNING_RATE = 1e-4

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 128
HIDDEN_DIM = 768

NUMBER_OF_LABELS = 2

model_path_prefix = model_folder_path + '/patch_classifier_variant_7_11112021_model_'
best_java_model_path = model_folder_path + '/patch_classifier_variant_7_best_java_11112021_model.sav'
best_python_model_path = model_folder_path + '/patch_classifier_variant_7_best_python_11112021_model.sav'

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Embedding') == -1):
        nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))


def predict_test_data(model, testing_generator, device):
    y_pred = []
    y_test = []
    probs = []
    test_ids = []
    with torch.no_grad():
        model.eval()
        for ids, url, before_batch, after_batch, label_batch in testing_generator:
            before_batch, after_batch, label_batch = before_batch.to(device), after_batch.to(device), label_batch.to(
                device)
            outs = model(before_batch, after_batch)

            outs = F.softmax(outs, dim=1)

            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            test_ids.extend(label_batch.tolist())

        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    return precision, recall, f1, auc


def find_max_length(arr):
    max_len = 0
    for elem in arr:
        if len(elem) > max_len:
            max_len = len(elem)
    return max_len


def custom_collate(batch):
    id, url, before, after, label = zip(*batch)

    # before: list embeddings of files
    max_before = find_max_length(before)
    # if max_before < 5:
    #     max_before = 5
    before_features = torch.zeros((len(batch), max_before, 768))

    max_after = find_max_length(after)
    # if max_after < 5:
    #     max_after = 5

    after_features = torch.zeros((len(batch), max_after, 768))
    for i in range(len(batch)):
        before = torch.FloatTensor(batch[i][2]).to(device)
        j, k = before.size(0), before.size(1)
        before_features[i] = torch.cat(
            [before,
             torch.zeros((max_before - j, k), device=device)])

    for i in range(len(batch)):
        after = torch.FloatTensor(batch[i][3]).to(device)
        j, k = after.size(0), after.size(1)
        after_features[i] = torch.cat(
            [after,
             torch.zeros((max_after - j, k), device=device)])

    label = torch.tensor(label).to(device)

    return id, url, before_features.float(), after_features.float(), label.long()


def train(model, training_generator, val_java_generator, val_python_generator, test_java_generator, test_python_generator):
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses, valid_losses = [], []
    best_auc_val_java = 0
    best_auc_val_python = 0

    print("Training...")
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        print("Calculating commit training loss...")
        current_batch = 0
        for ids, url, before_batch, after_batch, label_batch in training_generator:
            before_batch, after_batch, label_batch = before_batch.to(device), after_batch.to(device), label_batch.to(
                device)
            outs = model(before_batch, after_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))

        train_losses, valid_losses = [], []

        with torch.no_grad():
            model.eval()
            print("Result on Java validation dataset...")
            precision, recall, f1, auc = predict_test_data(model=model,
                                                           testing_generator=val_java_generator,
                                                           device=device)

            if auc > best_auc_val_java:
                best_auc_val_java = auc
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                               best_java_model_path)
                else:
                    torch.save(model.state_dict(), best_java_model_path)

            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))
            print("AUC: {}".format(auc))
            print("-" * 32)

            print("Result on Python validation dataset...")
            precision, recall, f1, auc = predict_test_data(model=model,
                                                           testing_generator=val_python_generator,
                                                           device=device)

            if auc > best_auc_val_python:
                best_auc_val_python = auc
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(),
                               best_python_model_path)
                else:
                    torch.save(model.state_dict(), best_python_model_path)

            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))
            print("AUC: {}".format(auc))
            print("-" * 32)

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(),
                           model_path_prefix + '_patch_classifier_epoc_' + str(epoch) + '.sav')
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


def do_train():
    print("Dataset name: {}".format(dataset_name))

    url_data, label_data = utils.get_data(dataset_name)
    train_ids, val_java_ids, val_python_ids, test_java_ids, test_python_ids = [], [], [], [], []

    index = 0
    id_to_url = {}
    id_to_label = {}

    print("Preparing data indices...")
    for i, url in enumerate(url_data['train']):
        train_ids.append(index)
        label = label_data['train'][i]
        id_to_url[index] = url
        id_to_label[index] = label
        index += 1

    for i, url in enumerate(url_data['val_java']):
        val_java_ids.append(index)
        label = label_data['val_java'][i]
        id_to_url[index] = url
        id_to_label[index] = label
        index += 1

    for i, url in enumerate(url_data['val_python']):
        val_python_ids.append(index)
        label = label_data['val_python'][i]
        id_to_url[index] = url
        id_to_label[index] = label
        index += 1

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        label = label_data['test_java'][i]
        id_to_url[index] = url
        id_to_label[index] = label
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        label = label_data['test_python'][i]
        id_to_url[index] = url
        id_to_label[index] = label
        index += 1


    print("Preparing dataset...")
    training_set = HunkDataset(train_ids, id_to_label, id_to_url)
    val_java_set = HunkDataset(val_java_ids, id_to_label, id_to_url)
    val_python_set = HunkDataset(val_python_ids, id_to_label, id_to_url)
    test_java_set = HunkDataset(test_java_ids, id_to_label, id_to_url)
    test_python_set = HunkDataset(test_python_ids, id_to_label, id_to_url)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS, collate_fn=custom_collate)
    val_java_generator = DataLoader(val_java_set, **VALIDATION_PARAMS, collate_fn=custom_collate)
    val_python_generator = DataLoader(val_python_set, **VALIDATION_PARAMS, collate_fn=custom_collate)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS, collate_fn=custom_collate)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS, collate_fn=custom_collate)

    model = PatchClassifierByHunk()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    for m in model.modules():
        weights_init(m)

    train(model=model, training_generator=training_generator, val_java_generator=val_java_generator,
          val_python_generator=val_python_generator, test_java_generator=test_java_generator, test_python_generator=test_python_generator)


if __name__ == '__main__':
    do_train()
