import json
import ensemble_classifier
import utils
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tqdm import tqdm
import csv

dataset_name = 'ase_dataset_sept_19_2021.csv'


def predict_from_neighbor_data(test_feature, neighbor_item, url_to_features):
    pos_features = [url_to_features[item[0]] for item in neighbor_item['pos']]
    neg_features = [url_to_features[item[0]] for item in neighbor_item['neg']]
    features = pos_features + neg_features
    labels = ([1] * len(pos_features)) + ([0] * len(neg_features))

    clf = LogisticRegression(random_state=109).fit(features, labels)
    pred_prob = clf.predict_proba([test_feature])[0][1]

    return pred_prob



def do_train():
    train_feature_path = [
        'features/feature_variant_1_train.txt',
        'features/feature_variant_2_train.txt',
        'features/feature_variant_3_train.txt',
        'features/feature_variant_5_train.txt',
        'features/feature_variant_6_train.txt',
        'features/feature_variant_7_train.txt',
        'features/feature_variant_8_train.txt'
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

    print("Reading data...")
    url_to_features = {}
    print("Reading train data")
    url_to_features.update(ensemble_classifier.read_feature_list(train_feature_path, reshape=True, need_list=True))

    print("Reading test java data")
    url_to_features.update(ensemble_classifier.read_feature_list(test_java_feature_path, reshape=True, need_list=True))

    print("Finish reading")
    url_data, label_data, url_to_pl, url_to_label = utils.get_data(dataset_name, need_pl=True)

    print("Loading neighbor data...")
    url_to_neighbor = json.load(open('url_to_neighbour_java.txt', 'r'))
    print("Finish loading neighbor data...")

    y_pred = []
    y_test = []
    urls = []
    for test_url, neighbor_item in tqdm(url_to_neighbor.items()):
        test_feature = url_to_features[test_url]
        test_label = url_to_label[test_url]
        pred_prob = predict_from_neighbor_data(test_feature, neighbor_item, url_to_features)
        y_pred.append(pred_prob)
        y_test.append(test_label)
        urls.append(test_url)
    auc = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
    print("AUC: {}".format(auc))

    with open('neighbour_ensemble_pred_prob.csv', 'w') as file:
        writer = csv.writer(file)
        for i, url in enumerate(urls):
            writer.writerow([url, y_pred[i], y_test[i]])


if __name__ == '__main__':
    do_train()
