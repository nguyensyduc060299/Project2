from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from  sklearn.svm import SVC
import numpy as np
def collect_data(datapath, type, tf_idf):
    lines = []
    with open(datapath,'r') as f:
        lines = f.read().splitlines()

    labels, docs = [], []
    for line in lines:
        labels.append(line.split('<fff>')[0])
        docs.append(line.split('<fff>')[-1])
    if type == 'train' :
        docs_vector = tf_idf.fit_transform(docs)
    else:
        docs_vector = tf_idf.transform(docs)
    return labels, docs_vector

def accuracy(predict, labels):
    labels = np.array(labels, dtype=int)
    predict = np.array(predict, dtype=int)
    right = np.equal(predict, labels)
    acc = np.sum(right.astype(float))/labels.size
    return acc

def LinearSVM(C, tf_idf):
    labels_train, docs_vector_train = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt','train', tf_idf)
    labels_test, docs_vector_test = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\test_data.txt', 'test', tf_idf)
    SVM = LinearSVC(
        C = C,
        tol = 1e4,
        verbose = False
    )

    SVM.fit(docs_vector_train,labels_train)
    predict = SVM.predict(docs_vector_test)
    return accuracy(predict, labels_test)

def KernelSVM(C, type, degree,gamma, coefo, tf_idf):
    labels_train, docs_vector_train = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt','train', tf_idf)
    labels_test, docs_vector_test = collect_data('E:\\20192\Project 2\\20news-bydate\\result\\test_data.txt', 'test',tf_idf)
    kernel = SVC(
        C = C,
        kernel= type,
        degree= degree,
        gamma = gamma,
        coef0= coefo
    )
    kernel.fit(docs_vector_train, labels_train)
    predict = kernel.predict(docs_vector_test)
    return accuracy(predict, labels_test)

def runSVM():
    tf_idf = TfidfVectorizer(min_df=6)
    train_linear_SVM = LinearSVM(1e4,tf_idf)
    train_soft_margin_SVM = LinearSVM(100,tf_idf)
    train_kernel_poly = KernelSVM(C = 100, type = 'poly', degree= 2, gamma=4, coefo= 1, tf_idf=tf_idf )
    train_kernel_sigmoid = KernelSVM(100, 'sigmoid',0,0.3,1, tf_idf)
    train_kernel_rbf = KernelSVM(100, 'rbf', 0, 0.5, 1, tf_idf)
    print("LinearSVM: " + str(train_linear_SVM))
    print("Soft-Margin SVM: "+ str(train_soft_margin_SVM))
    print("Kernel Poly SVM: " + str(train_kernel_poly))
    print("Kernel Sigmoid SVM: "+ str(train_kernel_sigmoid))
    print("Kernel RBF SVM: "+ str(train_kernel_rbf))

#-------------------------------------------------------------------------

runSVM()