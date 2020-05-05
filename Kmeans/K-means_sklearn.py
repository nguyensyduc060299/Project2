from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
def collect_data(datapath):
    lines = []
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()

    docs, labels = [], []
    for line in lines:
        docs.append(line.split("<fff>")[-1])
        labels.append(line.split("<fff>")[0])

    tf_idf = TfidfVectorizer(min_df=6)
    doc_vector = tf_idf.fit_transform(docs)

    return labels, doc_vector

def purity(labels, label_root):
    cluster_label = dict([(k, 0) for k in range(20)])
    for k in range(20):
        Xk = []
        for index in range(len(labels)):
            if labels[index] == k:
                Xk.append(index)

        cluster_label[k] = Xk
    cnt_doc = 0
    for k in range(20):

        Xk = cluster_label[k]
        print(len(Xk))
        count_max = dict([(str(label), 0) for label in range(20)])
        max = 0
        for index in range(len(Xk)):
            count_max[label_root[Xk[index]]] +=1
        for label in range(20):
            if count_max[str(label)] > max:
                max = count_max[str(label)]

        cnt_doc += max

    return (float(cnt_doc)/float(len(label_root)))

labels_train, docs_vector_train = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt')
kmeans = KMeans(n_clusters=20, init='k-means++',random_state=2018, n_init= 5).fit(docs_vector_train)
labels_trained = kmeans.predict(docs_vector_train)
print(labels_trained)
print(purity(labels_trained,labels_train))