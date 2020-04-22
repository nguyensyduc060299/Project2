from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
def LogisticRegressionSKLearn():
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

    tf_idfs = TfidfVectorizer(min_df=10)
    docs_train_vector = tf_idfs.fit_transform(docs_train)
    lg_sklearn = LogisticRegression(C=20, max_iter=250)
    lg_sklearn.fit(docs_train_vector, labels_train)
    docs_test_vector = tf_idfs.transform(docs_test)
    train_predict = lg_sklearn.predict(docs_train_vector)
    test_predict = lg_sklearn.predict(docs_test_vector)
    print(accuracy_score(train_predict, labels_train))
    print(accuracy_score(test_predict, labels_test))

LogisticRegressionSKLearn()