import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

def collect_data():
    path = 'E:\\20192\\Project 2\\20news-bydate\\20news-bydate-train'
    list_newsgroup = [newsgroup for newsgroup in os.listdir(path)]
    list_newsgroup.sort()

    def collect_data_from(datapath):
        data = []
        for id, group in enumerate(list_newsgroup):
            stt = id
            dir_path = datapath + '\\' + group
            list_files = os.listdir(dir_path)
            for file in list_files:
                file_patch = dir_path + '\\' + file
                with open(file_patch, 'r') as f:
                    text = f.read().lower()
                    words = [PorterStemmer().stem(word)
                             for word in re.split(r'\W+', text)
                             if word not in stopwords.words('english')]
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(stt) + '<fff>' + file + '<fff>' + content)
        return data

    train_data = collect_data_from('E:\\20192\\Project 2\\20news-bydate\\20news-bydate-train')
    test_data = collect_data_from('E:\\20192\\Project 2\\20news-bydate\\20news-bydate-test')
    full_data = train_data + test_data
    with open('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('E:\\20192\\Project 2\\20news-bydate\\result\\test_data.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('E:\\20192\\Project 2\\20news-bydate\\result\\full_data.txt', 'w') as f:
        f.write('\n'.join(full_data))

def get_if_idf(datapath):
    def cal_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size*1./df)

    with open(datapath) as f:
        lines = f.read().splitlines()

    doc_count = defaultdict(int)
    corpus_size = len(lines)

    for line in lines:
        item_line = line.split('<fff>')
        text = item_line[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1

    vocal_idfs = [(word, cal_idf(df, corpus_size)) for word, df in zip(doc_count.keys(),
                  doc_count.values()) if df > 10]

    print('Vocalbulary size : {}'.format(len(vocal_idfs)))

    with open('E:\\20192\\Project 2\\20news-bydate\\result\\vocalbulary.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in vocal_idfs]))

    word_IDs = dict([(word, index) for index, (word, idf) in enumerate(vocal_idfs)])
    idfs = dict(vocal_idfs)

    with open(datapath) as f:
        docs = [
            (int(line.split('<fff>')[0]), int(line.split('<fff>')[1]),
             line.split('<fff>')[2])
             for line in f.read().splitlines()]

    tf_idf = []
    for doc in docs:
        label, doc_id, text = doc
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(word))
        max_freq = max([words.count(word) for word in word_set])
        words_if_idfs = []
        for word in word_set:
            freq = words.count(word)
            if_idf_value = freq *1. / max_freq * idfs[word]
            words_if_idfs.append((word_IDs[word],if_idf_value))

        rep = ' '.join(words_if_idfs)
        tf_idf.append(label, doc_id, rep)

    with open('E:\\20192\\Project 2\\20news-bydate\\result\\vector.txt', 'w') as f:
        f.write('\n'.join(tf_idf))



collect_data()
get_if_idf('E:\\20192\Project 2\\20news-bydate\\result\\train_data.txt')









