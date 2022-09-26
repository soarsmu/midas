import ensemble_classifier
import utils
import csv
import json
from torch import nn
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

dataset_name = 'ase_dataset_sept_19_2021.csv'


cos = nn.CosineSimilarity(dim=0, eps=1e-6)


def calculate_similarity(test_feature, train_feature):
    return cos(test_feature, train_feature).item()


def find_neighbour(test_url, url_to_features, url_data, label_data, url_to_pl):
    test_feature = url_to_features[test_url]
    test_pl = url_to_pl[test_url]
    data = {}

    # store url and similarity score
    pos_list = []
    neg_list = []
    for i, url in enumerate(url_data['train']):
        if url_to_pl[url] != test_pl:
            continue

        if label_data['train'][i] == 1:
            pos_list.append((url, calculate_similarity(test_feature, url_to_features[url])))
        else:
            neg_list.append((url, calculate_similarity(test_feature, url_to_features[url])))

    pos_neighbour = sorted(pos_list, key=lambda x: x[1], reverse=True)[:10]
    neg_neighbour = sorted(neg_list, key=lambda x: x[1], reverse=True)[:10]

    data['pos'] = pos_neighbour
    data['neg'] = neg_neighbour

    return data


def process():
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
    url_to_features.update(ensemble_classifier.read_feature_list(train_feature_path, reshape=True))
    # print("Reading val data")
    # url_to_features.update(ensemble_classifier.read_feature_list(val_feature_path))
    print("Reading test java data")
    url_to_features.update(ensemble_classifier.read_feature_list(test_java_feature_path, reshape=True))
    # print("Reading test python data")
    # url_to_features.update(ensemble_classifier.read_feature_list(test_python_feature_path))

    print("Finish reading")
    url_data, label_data, url_to_pl, url_to_label = utils.get_data(dataset_name, need_pl=True)

    url_to_neighbor = {}

    count = 0
    for i, url in enumerate(url_data['test_java']):
        # if label_data['test_java'][i] == 0:
        #     continue

        count += 1
        if count % 100 == 0:
            print("finish: {}/{}".format(count, len(url_data['test_java'])))
        # print(url)
        url_to_neighbor[url] = find_neighbour(url, url_to_features, url_data, label_data, url_to_pl)
        # if count == 10:
        #     break
    json.dump(url_to_neighbor, open('url_to_neighbour_java.txt', 'w'))


def calculate_norm_and_dot():
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
    # print("Reading train data")
    # url_to_features.update(ensemble_classifier.read_feature_list(train_feature_path))
    print("Reading val data")
    url_to_features.update(ensemble_classifier.read_feature_list(val_feature_path, reshape=True))
    # print("Reading test java data")
    # url_to_features.update(ensemble_classifier.read_feature_list(test_java_feature_path))
    # print("Reading test python data")
    # url_to_features.update(ensemble_classifier.read_feature_list(test_python_feature_path))

    # calculate dot and norm preemptively
    data = {}
    norms = {}
    print("Calculating norm...")
    for url, features in tqdm(url_to_features.items()):
        norms[url] = norm(features)

    url_list = list(url_to_features.keys())

    print("Calculating dot...")
    dots = {}
    for i in tqdm(range(len(url_list))):
        for j in range(len(url_list)):
            if i < j:
                a = url_to_features[url_list[i]]
                b = url_to_features[url_list[j]]
                dots[url_list[i] + url_list[j]] = dot(a, b)

    data['norms'] = norms
    data['dots'] = dots

    json.dump(data, open('url_consine_data.txt', 'w'))


if __name__ == '__main__':
    process()
