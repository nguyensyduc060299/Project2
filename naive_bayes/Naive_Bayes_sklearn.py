
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def naive_bayes_sklearn():
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

    MuNB = MultinomialNB()
    MuNB.fit(docs_train_vector, labels_train)

    docs_test_vector = tf_idfs.transform(docs_test)

    result = MuNB.predict(docs_test_vector)

    cnt = 0
    for index in range(len(labels_test)):
        if result[index] == labels_test[index]:
            cnt+=1

    print("Correct: " + str(cnt) + '/' + str(len(result)))
    print("Accuracy: " + str(cnt / len(result)))


naive_bayes_sklearn()



