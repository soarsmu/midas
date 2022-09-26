# giang temporarily switch EnsembleModel to EnsembleModelFileLevelCNN
# for lineLSTM and lineGRU, just use EnsembleModel
# for hunk-level FCN, use EnsembleModelHunkLevelFCN

import os
import json
import utils
from torch.utils.data import DataLoader
from entities import EnsembleDataset, EnsemblePcaDataset
from model import EnsembleModel
import torch
from torch import cuda
from torch import nn as nn
from transformers import AdamW
from transformers import get_scheduler
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import csv
import argparse
from variant_ensemble import write_feature_to_file

directory = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'ase_dataset_sept_19_2021.csv'
# dataset_name = 'huawei_sub_dataset.csv'

FINAL_MODEL_PATH = None
JAVA_RESULT_PATH = None
PYTHON_RESULT_PATH = None

TRAIN_BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5
NUMBER_OF_EPOCHS = 20

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def write_prob_to_file(file_path, urls, probs):
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(urls):
            writer.writerow([url, probs[i]])


def read_features_from_file(file_path):
    file_path = os.path.join(directory, file_path)
    with open(file_path, 'r') as reader:
        data = json.loads(reader.read())

    return data


def read_feature_list(file_path_list, reshape=False, need_list=False, need_extend=False):
    url_to_feature = {}
    for file_path in file_path_list:
        data = read_features_from_file(file_path)
        for url, feature in data.items():
            if url not in url_to_feature:
                url_to_feature[url] = []
            if not need_extend:
                url_to_feature[url].append(feature)
            else:
                url_to_feature[url].extend(feature) 
    if not reshape:
        return url_to_feature
    else:
        url_to_combined = {}
        if reshape:
            for url in url_to_feature.keys():
                features = url_to_feature[url]
                combine = []
                for feature in features:
                    combine.extend(feature)
                if not need_list:
                    combine = torch.FloatTensor(combine)
                url_to_combined[url] = combine

        return url_to_combined




def predict_test_data(model, testing_generator, device, need_prob=False, need_features=False):
    y_pred = []
    y_test = []
    probs = []
    features = []
    urls = []
    with torch.no_grad():
        model.eval()
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(testing_generator):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)

            label_batch = label_batch.to(device)
                
            outs, pca_features = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, need_features=True)

            outs = F.softmax(outs, dim=1)

            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
            features.extend(list(pca_features.tolist()))

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
        return precision, recall, f1, auc, urls, probs, features


def train(model, learning_rate, number_of_epochs, training_generator, test_java_generator, test_python_generator):
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
        for ids, url_batch, feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8, label_batch in tqdm(training_generator):
            feature_1 = feature_1.to(device)
            feature_2 = feature_2.to(device)
            feature_3 = feature_3.to(device)
            feature_5 = feature_5.to(device)
            feature_6 = feature_6.to(device)
            feature_7 = feature_7.to(device)
            feature_8 = feature_8.to(device)

            label_batch = label_batch.to(device)

            outs = model(feature_1, feature_2, feature_3, feature_5, feature_6, feature_7, feature_8)
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



    print("Result on Java testing dataset...")
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model,
                                                testing_generator=test_java_generator,
                                                device=device, need_prob=True)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(JAVA_RESULT_PATH, urls, probs)

    print("Result on Python testing dataset...")
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model,
                                                testing_generator=test_python_generator,
                                                device=device, need_prob=True)
        
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)

    write_prob_to_file(PYTHON_RESULT_PATH, urls, probs)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    return model


def do_train(args):
    global FINAL_MODEL_PATH
    global JAVA_RESULT_PATH
    global PYTHON_RESULT_PATH

    FINAL_MODEL_PATH = args.model_path
    if FINAL_MODEL_PATH is None or FINAL_MODEL_PATH == '':
        raise Exception("Model path must not be None or empty")

    JAVA_RESULT_PATH = args.java_result_path
    if JAVA_RESULT_PATH is None or JAVA_RESULT_PATH == '':
        raise Exception("Java result path must not be None or empty")

    PYTHON_RESULT_PATH = args.python_result_path
    if PYTHON_RESULT_PATH is None or PYTHON_RESULT_PATH == '':
        raise Exception("Java result path must not be None or empty")

    variant_to_drop = []
    if args.variant_to_drop is not None:
        for variant in args.variant_to_drop:
            variant_to_drop.append(int(variant))

    train_feature_path = [
        'features/feature_variant_1_train.txt',
        'features/feature_variant_2_train.txt',
        'features/feature_variant_3_train.txt',
        'features/feature_variant_5_train.txt',
        'features/feature_variant_6_train.txt',
        'features/feature_variant_7_train.txt',
        'features/feature_variant_8_train.txt'
    ]

    val_feature_path = [
        'features/feature_variant_1_val.txt',
        'features/feature_variant_2_val.txt',
        'features/feature_variant_3_val.txt',
        'features/feature_variant_5_val.txt',
        'features/feature_variant_6_val.txt',
        'features/feature_variant_7_val.txt',
        'features/feature_variant_8_val.txt'
    ]

    test_java_feature_path = [
        'features/feature_variant_1_test_java.txt',
        'features/feature_variant_2_test_java.txt',
        'features/feature_variant_3_test_java.txt',
        'features/feature_variant_5_test_java.txt',
        'features/feature_variant_6_test_java.txt',
        'features/feature_variant_7_test_java.txt',
        'features/feature_variant_8_test_java.txt'
    ]

    test_python_feature_path = [
        'features/feature_variant_1_test_python.txt',
        'features/feature_variant_2_test_python.txt',
        'features/feature_variant_3_test_python.txt',
        'features/feature_variant_5_test_python.txt',
        'features/feature_variant_6_test_python.txt',
        'features/feature_variant_7_test_python.txt',
        'features/feature_variant_8_test_python.txt'
    ]

    print("Reading data...")
    url_to_features = {}
    print("Reading train data")
    url_to_features.update(read_feature_list(train_feature_path))
    print("Reading test java data")
    url_to_features.update(read_feature_list(test_java_feature_path))
    print("Reading test python data")
    url_to_features.update(read_feature_list(test_python_feature_path))

    print("Finish reading")
    url_data, label_data = utils.get_data(dataset_name)

    feature_data = {}
    feature_data['train'] = []
    feature_data['test_java'] = []
    feature_data['test_python'] = []

    for url in url_data['train']:
        feature_data['train'].append(url_to_features[url])

    for url in url_data['test_java']:
        feature_data['test_java'].append(url_to_features[url])

    for url in url_data['test_python']:
        feature_data['test_python'].append(url_to_features[url])

    val_ids, test_java_ids, test_python_ids = [], [], []
    index = 0
    id_to_url = {}
    id_to_label = {}
    id_to_feature = {}

    for i, url in enumerate(url_data['train']):
        val_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['train'][i]
        id_to_feature[index] = feature_data['train'][i]
        index += 1

    for i, url in enumerate(url_data['test_java']):
        test_java_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_java'][i]
        id_to_feature[index] = feature_data['test_java'][i]
        index += 1

    for i, url in enumerate(url_data['test_python']):
        test_python_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data['test_python'][i]
        id_to_feature[index] = feature_data['test_python'][i]
        index += 1

    training_set = EnsembleDataset(val_ids, id_to_label, id_to_url, id_to_feature)
    test_java_set = EnsembleDataset(test_java_ids, id_to_label, id_to_url, id_to_feature)
    test_python_set = EnsembleDataset(test_python_ids, id_to_label, id_to_url, id_to_feature)

    training_generator = DataLoader(training_set, **TRAIN_PARAMS)
    test_java_generator = DataLoader(test_java_set, **TEST_PARAMS)
    test_python_generator = DataLoader(test_python_set, **TEST_PARAMS)

    model = EnsembleModel(args.ablation_study, variant_to_drop)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    train(model=model,
          learning_rate=LEARNING_RATE,
          number_of_epochs=NUMBER_OF_EPOCHS,
          training_generator=training_generator,
          test_java_generator=test_java_generator,
          test_python_generator=test_python_generator)


def infer_dataset(model_path, partition, ablation_study, variant_to_drop, prob_path):
    # val_feature_path = [
    #     'features/feature_variant_1_val.txt',
    #     'features/feature_variant_2_val.txt',
    #     'features/feature_variant_3_val.txt',
    #     'features/feature_variant_5_val.txt',
    #     'features/feature_variant_6_val.txt',
    #     'features/feature_variant_7_val.txt',
    #     'features/feature_variant_8_val.txt'
    # ]


    test_java_feature_path = [
        'features/feature_variant_1_test_java.txt',
        'features/feature_variant_2_test_java.txt',
        'features/feature_variant_3_test_java.txt',
        'features/feature_variant_5_test_java.txt',
        'features/feature_variant_6_test_java.txt',
        'features/feature_variant_7_test_java.txt',
        'features/feature_variant_8_test_java.txt'
    ]

    test_python_feature_path = [
        'features/feature_variant_1_test_python.txt',
        'features/feature_variant_2_test_python.txt',
        'features/feature_variant_3_test_python.txt',
        'features/feature_variant_5_test_python.txt',
        'features/feature_variant_6_test_python.txt',
        'features/feature_variant_7_test_python.txt',
        'features/feature_variant_8_test_python.txt'
    ]

    # model = EnsembleModelHunkLevelFCN(False, [])

    model = EnsembleModel(ablation_study, variant_to_drop)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('model/patch_ensemble_model.sav'))
    model.to(device)

    print("Reading data")
    url_to_features = read_feature_list(test_python_feature_path)


    print("Finish reading")
    url_data, label_data = utils.get_data(dataset_name)

    feature_data = {}
    feature_data[partition] = []

    for url in url_data[partition]:
        feature_data[partition].append(url_to_features[url])

    val_ids = []
    index = 0
    id_to_url = {}
    id_to_label = {}
    id_to_feature = {}

    for i, url in enumerate(url_data[partition]):
        val_ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data[partition][i]
        id_to_feature[index] = feature_data[partition][i]
        index += 1

    val_set = EnsembleDataset(val_ids, id_to_label, id_to_url, id_to_feature)

    val_generator = DataLoader(val_set, **TEST_PARAMS)
    

    print("Result on dataset...")
    precision, recall, f1, auc, urls, probs, features = predict_test_data(model=model,
                                                    testing_generator=val_generator,
                                                    device=device, need_prob=True)



    write_prob_to_file(prob_path, urls, probs)
    # write_feature_to_file(feature_path, urls, features)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("AUC: {}".format(auc))
    print("-" * 32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Classifier')
    parser.add_argument('--ablation_study',
                        type=bool,
                        default=False,
                        help='Do ablation study or not')
    parser.add_argument('-v',
                        '--variant_to_drop',
                        action='append',
                        required=False,
                        help='Select index of variant to drop, 1, 2, 3, 5, 6, 7, 8')
    parser.add_argument('--model_path',
                        type=str,
                        help='IMPORTANT select path to save model')
    parser.add_argument('--java_result_path',
                       type=str,
                       help='path to save prediction for Java projects')
    parser.add_argument('--python_result_path',
                        type=str,
                        help='path to save prediction for Python projects')
    args = parser.parse_args()
    do_train(args)

    # infer_dataset('test_python', 'features/feature_ensemble_test_python.txt')