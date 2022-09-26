from fileinput import filename
from linecache import getlines
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
import pandas as pd
import json

dataset_name = 'big_vf.csv'
EMBEDDING_DIRECTORY = '../finetuned_embeddings/variant_8'


def get_code_version(diff, added_version):
    code = ''
    lines = diff.splitlines()
    for line in lines:
        mark = '+'
        if not added_version:
            mark = '-'
        if line.startswith(mark):
            line = line[1:].strip()
            if line.startswith(('//', '/**', '/*', '*', '*/', '#')) or line.strip() == '':
                continue
            code = code + line + '\n'

    return code

def line_empty(line):
    if line.strip() == '':
        return True
    else:
        return False


def get_line_from_code(code):
    lines = []
    for line in code.split('\n'):
        if not line_empty(line):
            lines.append(line)

    return lines

def get_code_line(diff):
    removed_lines, added_lines = [], []

    removed_code = get_code_version(diff, False)
    added_code = get_code_version(diff, True)

    removed_lines = get_line_from_code(removed_code)
    added_lines = get_line_from_code(added_code)
    return removed_lines, added_lines

    
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def visualize(features, labels, ids, path, lines):
    tsne = TSNE(n_components=2).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    X = [[tx[i], ty[i]] for i in range(len(tx))]

    kmeans = KMeans(n_clusters=5, random_state=109).fit(X)

    kmeans_labels = kmeans.labels_
    cluster_to_ids = {}
    for i, cluster in enumerate(kmeans_labels):
        if cluster not in cluster_to_ids:
            cluster_to_ids[cluster] = []
        cluster_to_ids[cluster].append(i)

    # for i in range(len(ids)):
    #     print("({} ; {};    {}) =>  {}".format(labels[i], ids[i], lines[i], kmeans_labels[i]))

    for cluster, cluster_ids in cluster_to_ids.items():
        for id in cluster_ids:
            print("({} ; {};    {}) =>  {}".format(labels[id], id, lines[id], cluster))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors_per_class = {0: "red", 1: "green"}
    label_to_plot_label = {0 : "removed lines", 1 : "added lines"}
    for label in colors_per_class:
        indices = [i for i, l in enumerate(labels) if l == label]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # color = np.array(colors_per_class[label], dtype=np.float) / 255

        ax.scatter(current_tx, current_ty, c=colors_per_class[label], label=label_to_plot_label[label])
    
    # for i in range(len(tx)):
    #     ax.annotate(ids[i], (tx[i], ty[i]))

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig(path)


# commit_url = 'https://github.com/apache/tomcat/commit/a6b1ebc246b91b854237e5aad3dfd2b5460ea282'
# parts = commit_url.split('/')
# print(parts)
# repo = parts[3] + '_' + parts[4]
# repo = repo.replace('/', '_')
# commit_id = 'e6aa166246d1734f4798a9e31f78842f4c85c28b'

# embedding_file_path = EMBEDDING_DIRECTORY + '/' + parts[3] + '_' + parts[4] + '_commit_' + parts[6] + '.txt'
# df = pd.read_csv(dataset_name)
# df = df[df.label == 1]
# commit_id = set()
# df = df[['commit_id', 'repo']]
# file_to_commit = {}
# file_set = set()
# for item in df.values.tolist():
#     commit_url = 'https://github.com/' + item[1] + '/commit/' + item[0]
#     file_name = item[1].replace('/', '_') + '_commit_' + item[0] + '.txt'
#     file_to_commit[file_name] = commit_url

# # print(file_set)
# id2commit = {}
# commit_list = []
# for file_name in os.listdir(EMBEDDING_DIRECTORY):
#     if file_name in file_to_commit:
#         with open(EMBEDDING_DIRECTORY + '/' + file_name , 'r') as reader:
#             data = json.loads(reader.read())

#             before = data['before']
#             after = data['after']
#             if len(before) + len(after) > 200:
#                 print(file_name)
#                 commit_list.append((file_name, file_to_commit[file_name]))

# with open('commit_list.csv', 'w') as file:
#     for file_name, commit in commit_list:
#         file.write(file_name + ',' + commit + '\n')


###############################
# save_items = []
# for item in items:
#     commit_id = item[0]
#     repo = item[1]
#     url = 'https://github.com/' + repo + '/commit/' + commit_id
#     if url in commit_list:
#         save_items.append(item)
#         print(url)

# df = pd.DataFrame(save_items, columns=['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename'])
# df.to_csv('big_vf.csv', index=False)
#################################

df = pd.read_csv('commit_list.csv')
items = df.values.tolist()
commit_list = []
file_list = []
for i in range(len(items)):
    # print(i)
    file_name = items[i][0]
    commit_url = items[i][1]
    commit_list.append(commit_url)
    file_list.append(file_name)
    # print(commit_url)

print("Reading dataset...")
df = pd.read_csv(dataset_name)
df = df[['commit_id', 'repo', 'partition', 'diff', 'label', 'PL', 'LOC_MOD', 'filename']]
items = df.to_numpy().tolist()

url_to_diff = {}

for item in items:
    commit_id = item[0]
    repo = item[1]
    url = 'https://github.com/' + repo + '/commit/' + commit_id
    diff = item[3]

    if url not in url_to_diff:
        url_to_diff[url] = ''

    url_to_diff[url] = url_to_diff[url] + diff + '\n'

url_to_removed_lines = {}
url_to_added_lines = {}
for url, diff in url_to_diff.items():
    removed_lines, added_lines = get_code_line(diff)
    url_to_removed_lines[url] = removed_lines
    url_to_added_lines[url] = added_lines

count = 0
for i in range(len(file_list)):
    file_name = file_list[i]
    url = commit_list[i]

    count += 1
    if count != 27:
        continue
    print(url)

    # if url in ['https://github.com/eclipse/rdf4j/commit/c7d59bd718881fb678ebdeba825b8eb832044e23',
    # 'https://github.com/bcgit/bc-java/commit/413b42f4d770456508585c830cfcde95f9b0e93b']:
    #     continue 

    with open(EMBEDDING_DIRECTORY + '/' + file_name , 'r') as reader:
        
        data = json.loads(reader.read())

        before = data['before']
        after = data['after']
        
        if len(before) > 0:
            for j in range(len(before) - 1, 0, -1):
                before[j] = before[j] + before[j-1] 
            before[0] = before[0] + ([0] * 768)

        # print(before)
        # break

        if len(after) > 0:
            for j in range(len(after) - 1):
                after[j] = after[j] + after[j + 1]
            after[len(after) - 1] = after[len(after) - 1] + ([0] * 768)

        assert len(before) == len(url_to_removed_lines[url])
        
        assert len(after) == len(url_to_added_lines[url])

        labels = []
        ids = []
        removed_lines = url_to_removed_lines[url]
        added_lines = url_to_added_lines[url]
        for j in range(len(before)):
            labels.append(0)
            ids.append(j)
        for j in range(len(after)):
            labels.append(1)
            ids.append(j)

        visualize(before + after, labels, ids, 'img/' + str(i) + '.png', removed_lines + added_lines)
    # print(i)
    # break