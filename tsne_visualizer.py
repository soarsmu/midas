import enum
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import ensemble_classifier
import utils

dataset_name = 'ase_dataset_sept_19_2021.csv'

# X = np.asarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# labels = np.asarray([0, 1, 1, 0])
# colors = ['red' if label == 0 else 'green' for label in labels]
# X_embedded = TSNE(n_components=2, init='random').fit_transform(X)

# print(X_embedded)
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
# # plt.show()
            
# plt.savefig('tnse_visualization.png')


def concat_features(feature_list):
    out = []
    for features in feature_list:
        for feature in features:
            out.append(feature)

    return out


def visualize():
    ensemble_test_java_feature_path = ['features/feature_ensemble_test_java.txt']
    ensemble_test_python_feature_path =  ['features/feature_ensemble_test_python.txt']

    test_java_feature_path = [
        # 'features/feature_variant_1_test_java.txt',
        # 'features/feature_variant_2_test_java.txt',
        'features/feature_variant_3_test_java.txt',
        # 'features/feature_variant_5_test_java.txt',
        # 'features/feature_variant_6_test_java.txt',
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

    print("Reading test data")
    url_to_features.update(ensemble_classifier.read_feature_list(ensemble_test_java_feature_path))

    url_data, label_data = utils.get_data(dataset_name)

    features = []
    labels = []

    for i, url in enumerate(url_data['test_java']):
        labels.append(label_data['test_java'][i])
        features.append(concat_features(url_to_features[url]))

    features = np.asarray(features)
    labels = np.asarray(labels)

    features_ = []
    labels_ = []
    for i, label in enumerate(labels):
        if label == 0:
            features_.append(features[i])
            labels_.append(label)

    for i, label in enumerate(labels):
        if label == 1:
            features_.append(features[i])
            labels_.append(label)


    colors = ['red' if label == 0 else 'green' for label in labels_]

    # Giang: Replace to PCA
    # X = TSNE(n_components=2, init='random').fit_transform(features_)

    pca = PCA(n_components=2)
    X = pca.fit_transform(features_)

    print(X.shape)
    # f = plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    # plt.show()
                
    plt.title("PCA visualization for ensemble classifier on java")
    plt.savefig('pca_visualization_ensemble_java.png')


if __name__ == '__main__':
    visualize()