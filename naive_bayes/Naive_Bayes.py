
import numpy as np
from collections import defaultdict


def collect_data(datapath):
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()

    labels,docs = [], []

    for line in lines:
        labels.append(line.split('<fff>')[0])
        docs.append(line.split('<fff>')[2])

    return labels,docs

def trainning():

    labels_train, docs_train = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt')
    vocalbulary = set()
    with open('E:\\20192\\Project 2\\20news-bydate\\result\\vocalbulary.txt', 'r') as f:
        vocalbulary = set(line.split('<fff>')[0] for line in f.read().splitlines())

    cnt_doc = defaultdict(int)
    cnt_word = defaultdict(int)
    total_word = defaultdict(int)
    label_set = set()
    for index in range(len(labels_train)):
        label, doc = labels_train[index], docs_train[index]
        label_set.add(label)
        cnt_doc[label] += 1
        for word in doc.split():
            if word in vocalbulary:
                total_word[label] += 1
                cnt_word[(label,word)] += 1

    label_px = defaultdict(int)
    word_label_px = defaultdict(int)
    for label in label_set:
        label_px[label] = np.log10(cnt_doc[label]/len(docs_train))
        for word in vocalbulary:
            word_label_px[(label,word)] = np.log10((cnt_word[(label,word)]+1)/(total_word[label]+len(vocalbulary)))

    with open('E:\\20192\\Project 2\\20news-bydate\\result\\px_label.txt', 'w') as f:
        f.write('\n'.join([str(label)+'<fff>'+str(px) for label, px in label_px.items()]))
    with open('E:\\20192\\Project 2\\20news-bydate\\result\\px_word_label.txt', 'w') as f:
        f.write('\n'.join([str(label)+'<fff>'+word+'<fff>'+str(px) for (label,word), px in word_label_px.items()]))

    labels_test, docs_test = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\test_data.txt')
    list_doc_test = []
    for doc in docs_test:
        list_doc_test.append(list(set([word for word in doc.split() if word in vocalbulary])))

    result = []
    for doc in list_doc_test:
        map , label_trained = -100000000000, 0
        for label in label_px.keys():
            px_doc = label_px[label]
            for word in doc:
                px_doc += word_label_px[(label,word)]

            if map < px_doc:
                map = px_doc
                label_trained = label



        result.append(label_trained)

    return result,labels_test


result, labels_test = [], []
result, labels_test = trainning()
cnt = 0
for index in range(len(result)):
    if labels_test[index] == result[index]:
        cnt+=1
print("Correct: " + str(cnt)+'/'+ str(len(result)))
print("Accuracy: "+ str(cnt/len(result)))






