from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
import numpy as np
import copy

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
    doc_vector = doc_vector.toarray()

    return labels, doc_vector

def init_center(X, K):
    return X[np.random.choice(X.shape[0],K, replace=False)]

def select_cluster(X, centers):
    distance = cdist(X, centers)
    return np.argmin(distance, axis=1)

def update_center(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xi = X[labels == k, :]
        centers[k,:] = np.mean(Xi, axis=0)

    return centers

def stop(centers, new_centers):
    return (set([tuple(x) for x in centers])) == (set([tuple(x) for x in new_centers]))


def Kmeans(X, K, max_loop):
    centers = init_center(X,K)
    labels = []
    loop = 0
    while True:
        new_label = select_cluster(X, centers)
        new_centers = update_center(X, new_label, K)
        if stop(centers, new_centers):
            break
        centers = copy.deepcopy(new_centers)
        labels = copy.deepcopy(new_label)
        loop+=1

    return centers,labels, loop

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


#-----------RUN-------------------

labels_train, docs_vector_train = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt')
#print(docs_vector_train)
centers, labels_trained, loop = Kmeans(docs_vector_train, 20, 100)
print(purity(labels_trained,labels_train))



