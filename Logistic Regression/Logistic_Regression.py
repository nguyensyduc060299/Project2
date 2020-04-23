from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import copy

def logistic_regression():
    #collect data---------------------------------
    lines_train, lines_test = [], []
    with open('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt', 'r') as f:
        lines_train = f.read().splitlines()
    with open('E:\\20192\Project 2\\20news-bydate\\result\\test_data.txt', 'r') as f:
        lines_test = f.read().splitlines()

    labels_train, docs_train = [], []
    for line in lines_train:
        labels_train.append(line.split('<fff>')[0])
        docs_train.append(line.split('<fff>')[2])

    labels_test, docs_test = [], []
    for line in lines_test:
        labels_test.append(line.split('<fff>')[0])
        docs_test.append(line.split('<fff>')[2])

    #transform data---------------------------------
    tf_idfs = TfidfVectorizer(min_df=6)
    docs_train_vector = tf_idfs.fit_transform(docs_train)
    docs_test_vector = tf_idfs.transform(docs_test)


    #loss function------------------------------------------------
    docs_train_vector = docs_train_vector.toarray()
    docs_train_vector = docs_train_vector.T
    docs_test_vector = docs_test_vector.toarray()
    docs_test_vector = docs_test_vector.T
    def sigmoid(s):
        return 1/(1+np.exp(-s))

    list_label = list(set(labels_train))
    number_label = len(list_label)
    number_atr = docs_train_vector.shape[0]
    number_doc = docs_train_vector.shape[1]
    w = np.array([])
    max_count = 100
    learning_rate = 0.05
    tol = 1e-4
    check_w_after = 10
    for label in range(number_label):
        count = 0
        Y = []
        for c in labels_train:
            if str(label) == c:
                Y.append(1)
            else:
                Y.append(0)
        wi = np.random.randn(number_atr,1)
        while count < max_count:
            index = np.random.permutation(number_doc)
            for i in index:
                xi = docs_train_vector[:,i].reshape(number_atr,1)
                yi = Y[i]
                z = np.dot(wi.T,xi)
                zi = sigmoid(z)
                w_new = wi - learning_rate*(zi-yi)*xi
                if count % check_w_after == 0:
                    if np.linalg.norm(w_new - wi) < tol:
                       wi = copy.deepcopy(w_new)
                    break
                wi = copy.deepcopy(w_new)
            count += 1
        wi = wi.reshape(1,number_atr)
        w = np.append(w,[wi])
    w = w.reshape(number_label,number_atr)

    #trainning------------------------------------------------------------
    labels_trained = []
    for i in range(number_doc):
        xi = docs_train_vector[:,i].reshape(number_atr,1)
        y_pre = sigmoid(np.dot(w,xi))
        tmp = 0
        max_label = 0
        for index in range(number_label):
            if y_pre[index] > tmp:
                tmp = y_pre[index]
                max_label = index
        labels_trained.append(max_label)
    cnt_label = 0
    Y=[]
    for c in range(number_doc):
        Y.append(int(labels_train[c]))
    for i in range(number_doc):
        if labels_trained[i] == Y[i]:
            cnt_label +=1
    print("Train: "+str(cnt_label/number_doc))

    number_doc_test = docs_test_vector.shape[1]
    labels_tested = []
    for i in range(number_doc_test):
        xi = docs_test_vector[:, i].reshape(number_atr, 1)
        y_pre = sigmoid(np.dot(w, xi))
        tmp = 0
        max_label = 0
        for index in range(number_label):
            if y_pre[index] > tmp:
                tmp = y_pre[index]
                max_label = index
        labels_tested.append(max_label)
    cnt_label_test = 0
    Y_test = []
    for c in range(number_doc_test):
        Y_test.append(int(labels_test[c]))
    for i in range(number_doc):
        if labels_tested[i] == Y_test[i]:
            cnt_label_test += 1
    print("Test: " + str(cnt_label_test / number_doc_test))

#------------------------
logistic_regression()
