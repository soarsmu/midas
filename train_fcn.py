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
from entities import VariantSixDataset
from model import PatchClassifier
import pandas as pd
from tqdm import tqdm
import utils

dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset.csv'
directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')

NUMBER_OF_EPOCHS = 100

TRAIN_BATCH_SIZE = 512
VALIDATION_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-4

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

false_cases = []
CODE_LENGTH = 512
HIDDEN_DIM = 768

NUMBER_OF_LABELS = 2


model_path_prefix = model_folder_path + '/patch_classifier_variant_6_best_java_11112021_model_'
best_java_model_path = model_folder_path + '/patch_classifier_variant_6_best_java_11112021_model.sav'
best_python_model_path = model_folder_path + '/patch_classifier_variant_6_best_python_11112021_model.sav'

def get_data():
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)
    df = df[['commit_id', 'repo', 'partition', 'PL', 'label']]
    items = df.to_numpy().tolist()

    label_train = []
    label_val = []
    label_test_java = []
    label_test_python = []
    url_train = []
    url_val = []
    url_test_java = []
    url_test_python = []

    for item in tqdm(items):
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[2]
        pl = item[3]
        label = item[4]

        if partition == 'train':
            if url not in url_train:
                url_train.append(url)
                label_train.append(label)
        elif partition == 'val':
            if url not in url_val:
                url_val.append(url)
                label_val.append(label)
        elif partition == 'test':
            if pl == 'java':
                if url not in url_test_java:
                    url_test_java.append(url)
                    label_test_java.append(label)
            else:
                if url not in url_test_python:
                    url_test_python.append(url)
                    label_test_python.append(label)
        else:
            Exception("Invalid partition: {}".format(partition))

    print("Finish reading dataset")
    return url_train, url_val, url_test_java, url_test_python, label_train, label_val, label_test_java, label_test_python


def predict_test_data(model, testing_generator, device):
    y_pred = []
    y_test = []
    probs = []
    test_ids = []
    with torch.no_grad():
        model.eval()
        for ids, url, before_batch, after_batch, label_batch in testing_generator:
            before_batch, after_batch, label_batch = before_batch.to(device), after_batch.to(device), label_batch.to(device)
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
            auc_pr = metrics.average_precision_score(y_score=probs, y_true=y_test)
        except Exception:
            auc = 0
            auc_pr = 0

    print("Finish testing")
    return precision, recall, f1, auc


def train(model, training_generator, val_java_generator, val_python_generator, test_java_generator, test_python_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUMBER_OF_EPOCHS * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses, valid_losses = [], []
    print("Training...")
    best_auc_val_java = 0
    best_auc_val_python = 0
    for epoch in range(NUMBER_OF_EPOCHS):
        model.train()
        print("Calculating commit training loss...")
        current_batch = 0
        for ids, url, before_batch, after_batch, label_batch in training_generator:
            before_batch, after_batch, label_batch = before_batch.to(device), after_batch.to(device), label_batch.to(device)
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
    training_set = VariantSixDataset(train_ids, id_to_label, id_to_url)
    val_java_set = VariantSixDataset(val_java_ids, id_to_label, id_to_url)
    val_python_set = VariantSixDataset(val_python_ids, id_to_label, id_to_url)
    test_java_set = VariantSixDataset(test_java_ids, id_to_label, id_to_url)
    test_python_set = VariantSixDataset(test_python_ids, id_to_label, id_to_url)

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

    train(model=model, training_generator=training_generator, val_java_generator=val_java_generator, val_python_generator=val_python_generator,
          test_java_generator=test_java_generator, test_python_generator=test_python_generator)


if __name__ == '__main__':
    do_train()