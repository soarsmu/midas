import torch
from torch import nn as nn
import os
import csv
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import pandas as pd
from entities import VariantOneDataset, VariantTwoDataset, VariantFiveDataset, VariantSixDataset, VariantThreeDataset, \
    VariantSevenDataset, VariantEightDataset, VariantThreeFcnDataset
from model import VariantOneClassifier, VariantTwoClassifier, VariantFiveClassifier, VariantSixClassifier, \
    VariantThreeClassifier, VariantSevenClassifier, VariantEightClassifier, VariantEightLstmClassifier, VariantEightGruClassifier
import utils
import variant_1
import variant_2
import variant_3
import variant_5
import variant_6
import variant_7
import variant_8
import variant_8_lstm
import variant_8_gru
import variant_3_fcn
import variant_7_fcn
from sklearn import metrics
from statistics import mean
from sklearn.linear_model import LogisticRegression
import csv
import json

# dataset_name = 'huawei_csv_subset_slicing_limited_10.csv'
# dataset_name = 'huawei_sub_dataset.csv'
dataset_name = 'ase_dataset_sept_19_2021.csv'

directory = os.path.dirname(os.path.abspath(__file__))

model_folder_path = os.path.join(directory, 'model')


VARIANT_ONE_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_1'
VARIANT_TWO_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_2'
VARIANT_THREE_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_3'
VARIANT_FOUR_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_4'
VARIANT_FIVE_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_5'
VARIANT_SIX_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_6'
VARIANT_SEVEN_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_7'
VARIANT_EIGHT_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_8'
VARIANT_TWO_CNN_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_2_cnn'
VARIANT_SIX_CNN_EMBEDDINGS_DIRECTORY = '../finetuned_embeddings/variant_6_cnn'


VARIANT_ONE_MODEL_PATH = 'model/patch_variant_1_finetune_1_epoch_best_model.sav'
VARIANT_TWO_MODEL_PATH = 'model/patch_variant_2_finetune_1_epoch_best_model.sav'
VARIANT_THREE_MODEL_PATH = 'model/patch_variant_3_finetune_1_epoch_best_model.sav'
VARIANT_FIVE_MODEL_PATH = 'model/patch_variant_5_finetune_1_epoch_best_model.sav'
VARIANT_SIX_MODEL_PATH = 'model/patch_variant_6_finetune_1_epoch_best_model.sav'
VARIANT_SEVEN_MODEL_PATH = 'model/patch_variant_7_finetune_1_epoch_best_model.sav'
VARIANT_EIGHT_MODEL_PATH = 'model/patch_variant_8_finetune_1_epoch_best_model.sav'

VARIANT_TWO_CNN_MODEL_PATH = 'model/patch_variant_2_cnn_best_model.sav'
VARIANT_SIX_CNN_MODEL_PATH = 'model/patch_variant_6_cnn_best_model.sav'
VARIANT_EIGHT_LSTM_MODEL_PATH = 'model/patch_variant_8_lstm_model.sav'
VARIANT_EIGHT_GRU_MODEL_PATH = 'model/patch_variant_8_gru_model.sav'

VARIANT_THREE_FCN_MODEL_PATH = 'model/patch_variant_3_fcn.sav'
VARIANT_SEVEN_FCN_MODEL_PATH = 'model/patch_variant_7_fcn.sav'

TEST_BATCH_SIZE = 64

TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
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


def write_feature_to_file(file_path, urls, features):
    file_path = os.path.join(directory, file_path)
    data = {}
    for i, url in enumerate(urls):
        data[url] = features[i]

    json.dump(data, open(file_path, 'w'))


def infer_variant_1(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantOneClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(VARIANT_ONE_MODEL_PATH))
    model.to(device)

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantOneDataset(ids, id_to_label, id_to_url, VARIANT_ONE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        auc, urls, features = variant_1.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)

    else:
        precision, recall, f1, auc, urls, probs = variant_1.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_2(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantTwoClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_TWO_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantTwoDataset(ids, id_to_label, id_to_url, VARIANT_TWO_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        auc, urls, features = variant_2.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_2.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_3_fcn(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantTwoClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_THREE_FCN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantThreeFcnDataset(ids, id_to_label, id_to_url, VARIANT_THREE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        auc, urls, features = variant_3_fcn.predict_test_data(model, generator, device, need_prob=True, need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_3_fcn.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_2_cnn(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantThreeClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_TWO_CNN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantThreeDataset(ids, id_to_label, id_to_url, VARIANT_TWO_CNN_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_3.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_3.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_3.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)



def infer_variant_3(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantThreeClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_THREE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantThreeDataset(ids, id_to_label, id_to_url, VARIANT_THREE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_3.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_3.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_3.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_5(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantFiveClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_FIVE_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantFiveDataset(ids, id_to_label, id_to_url, VARIANT_FIVE_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        auc, urls, features = variant_5.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_5.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_6(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantSixClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SIX_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSixDataset(ids, id_to_label, id_to_url, VARIANT_SIX_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        auc, urls, features = variant_6.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_6.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_7_fcn(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))
    model = VariantSixClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SEVEN_FCN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSixDataset(ids, id_to_label, id_to_url, VARIANT_SEVEN_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS)

    if need_feature_only:
        auc, urls, features = variant_7_fcn.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_7_fcn.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_6_cnn(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantSevenClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SIX_CNN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSevenDataset(ids, id_to_label, id_to_url, VARIANT_SIX_CNN_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_7.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_7.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_7.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_7(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantSevenClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_SEVEN_MODEL_PATH))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantSevenDataset(ids, id_to_label, id_to_url, VARIANT_SEVEN_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_7.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_7.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_7.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_8(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantEightClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_EIGHT_MODEL_PATH, map_location={'cuda:0': 'cuda:1'}))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantEightDataset(ids, id_to_label, id_to_url, VARIANT_EIGHT_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_8.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_8.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_8.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_8_lstm(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantEightLstmClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_EIGHT_LSTM_MODEL_PATH, map_location={'cuda:0': 'cuda:1'}))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantEightDataset(ids, id_to_label, id_to_url, VARIANT_EIGHT_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_8_lstm.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_8_lstm.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_8_lstm.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs)


def infer_variant_8_gru(partition, result_file_path, need_feature_only=False):
    print("Testing on partition: {}".format(partition))
    print("Saving result to: {}".format(result_file_path))

    model = VariantEightGruClassifier()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # model = nn.DataParallel(model)

    model.to(device)

    model.load_state_dict(torch.load(VARIANT_EIGHT_GRU_MODEL_PATH, map_location={'cuda:0': 'cuda:1'}))

    ids, id_to_label, id_to_url = get_dataset_info(partition)
    dataset = VariantEightDataset(ids, id_to_label, id_to_url, VARIANT_EIGHT_EMBEDDINGS_DIRECTORY)
    generator = DataLoader(dataset, **TEST_PARAMS, collate_fn=variant_8_gru.custom_collate)

    if need_feature_only:
        auc, urls, features = variant_8_gru.predict_test_data(model, generator, device, need_prob=True,
                                                          need_feature_only=need_feature_only)
        print("AUC: {}".format(auc))
        write_feature_to_file(result_file_path, urls, features)
    else:
        precision, recall, f1, auc, urls, probs = variant_8_gru.predict_test_data(model, generator, device, need_prob=True)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("-" * 32)

        write_prob_to_file(result_file_path, urls, probs) 
        

def get_dataset_info(partition):
    url_data, label_data = utils.get_data(dataset_name)
    ids = []
    index = 0
    id_to_url = {}
    id_to_label = {}

    for i, url in enumerate(url_data[partition]):
        ids.append(index)
        id_to_url[index] = url
        id_to_label[index] = label_data[partition][i]
        index += 1

    return ids, id_to_label, id_to_url


def read_pred_prob(file_path):
    df = pd.read_csv(file_path, header=None)
    url_to_prob = {}

    for url, prob in df.values.tolist():
        url_to_prob[url] = prob

    return url_to_prob


def get_auc_max_ensemble():
    print("Reading result...")
    variant_1_result = read_pred_prob('probs/prob_variant_1_finetune_1_epoch_test_python.txt')
    variant_2_result = read_pred_prob('probs/prob_variant_2_finetune_1_epoch_test_python.txt')
    variant_3_result = read_pred_prob('probs/prob_variant_3_finetune_1_epoch_test_python.txt')
    variant_5_result = read_pred_prob('probs/prob_variant_5_finetune_1_epoch_test_python.txt')
    variant_6_result = read_pred_prob('probs/prob_variant_6_finetune_1_epoch_test_python.txt')
    variant_7_result = read_pred_prob('probs/prob_variant_7_finetune_1_epoch_test_python.txt')
    variant_8_result = read_pred_prob('probs/prob_variant_8_finetune_1_epoch_test_python.txt')

    print("Finish reading result")

    url_to_max_prob = {}

    for url, prob_1 in variant_1_result.items():
        prob_2 = variant_2_result[url]
        prob_3 = variant_3_result[url]
        prob_5 = variant_5_result[url]
        prob_6 = variant_6_result[url]
        prob_7 = variant_7_result[url]
        prob_8 = variant_8_result[url]
        # url_to_max_prob[url] = mean([prob_1, prob_2, prob_3, prob_8])
        url_to_max_prob[url] = mean([prob_1, prob_2, prob_3, prob_5, prob_6, prob_7, prob_8])

    url_data, label_data = utils.get_data(dataset_name)
    url_test = url_data['test_python']
    label_test = label_data['test_python']

    y_score = []
    y_true = []
    for i, url in enumerate(url_test):
        y_true.append(label_test[i])
        y_score.append(url_to_max_prob[url])

    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)

    print("AUC: {}".format(auc))

    with open('probs/mean_prob_python.txt', 'w') as file:
        writer = csv.writer(file)
        for url, prob in url_to_max_prob.items():
            writer.writerow([url, prob])


def get_data_ensemble_model(prob_list, label_list):
    clf = LogisticRegression(random_state=109).fit(prob_list, label_list)

    return clf


def get_variant_result(variant_result_path):
    result = read_pred_prob(variant_result_path)

    return result


def get_prob(result_list, url):
    return [result[url] for result in result_list]


def get_partition_prob_list(result_path_list, partition):
    result_list = []
    for result_path in result_path_list:
        variant_result = get_variant_result(result_path)
        result_list.append(variant_result)

    url_data, label_data = utils.get_data(dataset_name)

    prob_list, label_list = [], []

    for i, url in enumerate(url_data[partition]):
        prob_list.append(get_prob(result_list, url))
        label_list.append(label_data[partition][i])

    return prob_list, label_list, url_data[partition]


def get_combined_ensemble_model():
    train_result_path_list = [
        ['variant_1_prob_train_java.txt', 'variant_1_prob_train_python.txt'],
        ['variant_2_prob_train_java.txt', 'variant_2_prob_train_python.txt'],
        ['variant_3_prob_train_java.txt', 'variant_3_prob_train_python.txt'],
        ['variant_5_prob_train_java.txt', 'variant_5_prob_train_python.txt'],
        ['variant_6_prob_train_java.txt', 'variant_6_prob_train_python.txt'],
        ['variant_7_prob_train_java.txt', 'variant_7_prob_train_python.txt']
    ]

    val_result_path_list = ['probs/prob_variant_1_finetune_1_epoch_val.txt',
                            'probs/prob_variant_2_finetune_1_epoch_val.txt',
                            'probs/prob_variant_3_finetune_1_epoch_val.txt',
                            'probs/prob_variant_5_finetune_1_epoch_val.txt',
                            'probs/prob_variant_6_finetune_1_epoch_val.txt',
                            'probs/prob_variant_7_finetune_1_epoch_val.txt',
                            'probs/prob_variant_8_finetune_1_epoch_val.txt']
                            # 'variant_7_prob_val.txt']

    test_java_result_path_list = ['probs/prob_variant_1_finetune_1_epoch_test_java.txt',
                                  'probs/prob_variant_2_finetune_1_epoch_test_java.txt',
                                  'probs/prob_variant_3_finetune_1_epoch_test_java.txt',
                                  'probs/prob_variant_5_finetune_1_epoch_test_java.txt',
                                  'probs/prob_variant_6_finetune_1_epoch_test_java.txt',
                                  'probs/prob_variant_7_finetune_1_epoch_test_java.txt',
                                  'probs/prob_variant_8_finetune_1_epoch_test_java.txt']
                                  # 'variant_7_prob_java.txt']

    test_python_result_path_list = ['probs/prob_variant_1_finetune_1_epoch_test_python.txt',
                                    'probs/prob_variant_2_finetune_1_epoch_test_python.txt',
                                    'probs/prob_variant_3_finetune_1_epoch_test_python.txt',
                                    'probs/prob_variant_5_finetune_1_epoch_test_python.txt',
                                    'probs/prob_variant_6_finetune_1_epoch_test_python.txt',
                                    'probs/prob_variant_7_finetune_1_epoch_test_python.txt',
                                    'probs/prob_variant_8_finetune_1_epoch_test_python.txt']
                                    # 'variant_7_prob_python.txt']

    # train_prob_list, train_label_list = get_partition_prob_list(train_result_path_list, 'train')
    val_prob_list, val_label_list, val_url_list = get_partition_prob_list(val_result_path_list, 'val')
    java_test_prob_list, java_test_label_list, java_test_url_list = get_partition_prob_list(test_java_result_path_list, 'test_java')
    python_test_prob_list, python_test_label_list, python_test_url_list = get_partition_prob_list(test_python_result_path_list, 'test_python')

    # train_ensemble_model = get_data_ensemble_model(train_prob_list, train_label_list)
    print("Training ensemble model...")
    val_ensemble_model = get_data_ensemble_model(val_prob_list, val_label_list)
    print("Finish training")

    print("Calculate AUC on Java...")
    y_probs = val_ensemble_model.predict_proba(java_test_prob_list)
    y_probs = [prob[1] for prob in y_probs.tolist()]
    auc = metrics.roc_auc_score(y_true=java_test_label_list, y_score=y_probs)
    print("AUC on Java of ensemble model: {}".format(auc))

    with open('probs/ensemble_prob_java.txt', 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(java_test_url_list):
            writer.writerow([url, y_probs[i]])

    print("Calculate AUC on Python...")
    y_probs = val_ensemble_model.predict_proba(python_test_prob_list)
    y_probs = [prob[1] for prob in y_probs.tolist()]
    auc = metrics.roc_auc_score(y_true=python_test_label_list, y_score=y_probs)
    print("AUC on Python of ensemble model: {}".format(auc))

    with open('probs/ensemble_prob_python.txt', 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(python_test_url_list):
            writer.writerow([url, y_probs[i]])


if __name__ == '__main__':
    # print("Inferring variant 1...")
    # infer_variant_1('train', 'features/feature_variant_1_train.txt', need_feature_only=True)
    # infer_variant_1('test_java', 'features/feature_variant_1_test_java.txt', need_feature_only=True)
    # infer_variant_1('test_python', 'features/feature_variant_1_test_python.txt', need_feature_only=True)
    # print('-' * 64)
    #
    # print("Inferring variant 2...")
    # infer_variant_2('train', 'features/feature_variant_2_train.txt', need_feature_only=True)
    # infer_variant_2('test_java', 'features/feature_variant_2_test_java.txt', need_feature_only=True)
    # infer_variant_2('test_python', 'features/feature_variant_2_test_python.txt', need_feature_only=True)
    # print('-' * 64)
    #

    #
    # print("Inferring variant 5...")
    # infer_variant_5('train', 'features/feature_variant_5_train.txt', need_feature_only=True)
    # infer_variant_5('test_java', 'features/prob_variant_5_test_java.txt', need_feature_only=True)
    # infer_variant_5('test_python', 'features/prob_variant_5_test_python.txt', need_feature_only=True)
    # print('-' * 64)
    #
    # print("Inferring variant 6...")
    # infer_variant_6('train', 'features/feature_variant_6_train.txt', need_feature_only=True)
    # infer_variant_6('test_java', 'features/feature_variant_6_test_java.txt', need_feature_only=True)
    # infer_variant_6('test_python', 'features/feature_variant_6_test_python.txt', need_feature_only=True)
    #
    # print("Inferring variant 7...")
    # infer_variant_7('val', 'features/feature_variant_7_val.txt', need_feature_only=True)
    # infer_variant_7('test_java', 'features/feature_variant_7_test_java.txt', need_feature_only=True)
    # infer_variant_7('test_python', 'features/feature_variant_7_test_python.txt', need_feature_only=True)

    # print("Inferring variant 8...")
    # infer_variant_8('train', 'features/feature_variant_8_train.txt', need_feature_only=True)
    # infer_variant_8('test_java', 'features/feature_variant_8_test_java.txt', need_feature_only=True)
    # infer_variant_8('test_python', 'features/feature_variant_8_test_python.txt', need_feature_only=True)

    # print("Inferring variant 3...")
    # infer_variant_3('train', 'features/feature_variant_3_train.txt', need_feature_only=True)
    # infer_variant_3('test_java', 'probs/prob_variant_3_finetune_1_epoch_test_java.txt', need_feature_only=False)
    # infer_variant_3('test_python', 'probs/prob_variant_3_finetune_1_epoch_test_python.txt', need_feature_only=False)

    # print("Inferring variant 2 CNN...")
    # infer_variant_2_cnn('train', 'features/feature_variant_2_cnn_train.txt', need_feature_only=True)
    # infer_variant_2_cnn('test_java', 'features/feature_variant_2_cnn_test_java.txt', need_feature_only=True)
    # infer_variant_2_cnn('test_python', 'features/feature_variant_2_cnn_test_python.txt', need_feature_only=True)


    # print("Inferring variant 6 CNN...")
    # infer_variant_6_cnn('train', 'features/feature_variant_6_cnn_train.txt', need_feature_only=True)
    # infer_variant_6_cnn('test_java', 'features/feature_variant_6_cnn_test_java.txt', need_feature_only=True)
    # infer_variant_6_cnn('test_python', 'features/feature_variant_6_cnn_test_python.txt', need_feature_only=True)

    # print("Inferring variant 8 LSTM...")
    # infer_variant_8_lstm('train', 'features/feature_variant_8_lstm_train.txt', need_feature_only=True)
    # infer_variant_8_lstm('test_java', 'features/feature_variant_8_lstm_test_java.txt', need_feature_only=True)
    # infer_variant_8_lstm('test_python', 'features/feature_variant_8_lstm_test_python.txt', need_feature_only=True)

    # print("Inferring variant 8 GRU...")
    # infer_variant_8_gru('train', 'features/feature_variant_8_gru_train.txt', need_feature_only=True)
    # infer_variant_8_gru('test_java', 'features/feature_variant_8_gru_test_java.txt', need_feature_only=True)
    # infer_variant_8_gru('test_python', 'features/feature_variant_8_gru_test_python.txt', need_feature_only=True)


    print("Inferring variant 3 FCN...")
    infer_variant_3_fcn('train', 'features/feature_variant_3_fcn_train.txt', need_feature_only=True)
    infer_variant_3_fcn('test_java', 'features/feature_variant_3_fcn_test_java.txt', need_feature_only=True)
    infer_variant_3_fcn('test_python', 'features/feature_variant_3_fcn_test_python.txt', need_feature_only=True)

    print("Inferring variant 7 FCN...")
    infer_variant_7_fcn('train', 'features/feature_variant_7_fcn_train.txt', need_feature_only=True)
    infer_variant_7_fcn('test_java', 'features/feature_variant_7_fcn_test_java.txt', need_feature_only=True)
    infer_variant_7_fcn('test_python', 'features/feature_variant_7_fcn_test_python.txt', need_feature_only=True)