import os
from multiprocessing import Process
from random import randint, uniform, choice

import fasttext
import time


def random_string(length):
    return ''.join(choice('0123456789ABCDEF') for i in range(length))


# download pretrain wordvector form here
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
# input_file             training file path (required)
# output                 output file path (required)
# label_prefix           label prefix ['__label__']
# lr                     learning rate [0.1]
# lr_update_rate         change the rate of updates for the learning rate [100]
# dim                    size of word vectors [100]
# ws                     size of the context window [5]
# epoch                  number of epochs [5]
# min_count              minimal number of word occurences [1]
# neg                    number of negatives sampled [5]
# word_ngrams            max length of word ngram [1]
# loss                   loss function {ns, hs, softmax} [softmax]
# bucket                 number of buckets [0]
# minn                   min length of char ngram [0]
# maxn                   max length of char ngram [0]
# thread                 number of threads [12]
# t                      sampling threshold [0.0001]
# silent                 disable the log output from the C++ extension [1]
# encoding               specify input_file encoding [utf-8]
# pretrained_vectors     pretrained word vectors (.vec file) for supervised learning []
def train(save_dri="./", epoch=300, word_ngrams=7, loss='hs', lr=0.01, lr_update_rate=10, dim=300, bucket=200000):
    classifier = fasttext.supervised('./training_data/taipei/training.data',
                                     save_dri + 'model_' + str(int(time.time())) + random_string(5),
                                     label_prefix='__label__', silent=False,
                                     lr=lr, lr_update_rate=lr_update_rate, epoch=epoch, dim=dim, bucket=bucket,
                                     word_ngrams=word_ngrams, loss=loss)
    result = classifier.test('./training_data/taipei/testing.data')
    print(result.precision)
    print(result.recall)
    print(result.nexamples)
    return classifier


def test_from_data():
    classifier = fasttext.load_model('model.bin', label_prefix='__label__')
    result = classifier.test('./training_data/taipei/testing.data')
    print(result.precision)
    print(result.recall)
    print(result.nexamples)


def test_result(classifiers):
    while (True):
        input_sentence = ''
        for j in input(">>> Input: "):
            input_sentence += j + " "
        print(input_sentence)
        for classifier in classifiers:
            prob = classifier.predict_proba([input_sentence])
            ans = (classifier.predict([input_sentence]))
            print(prob)
            print(ans)


def train_classifiers(model_num=10, save_dri="./" + str(int(time.time())) + "/"):
    if not os.path.exists(save_dri):
        os.makedirs(save_dri)
    loss_method = ['ns', 'hs', 'softmax']
    for i in range(model_num):
        epoch = randint(5, 300)
        word_ngrams = randint(1, 12)
        lr = uniform(0.01, 1)
        lr_update_rate = randint(5, 500)
        loss = loss_method[randint(0, 2)]
        dim = randint(100, 700)
        bucket = randint(5000, 500000)
        Process(target=train, kwargs={'save_dri': save_dri, 'epoch': epoch, 'word_ngrams': word_ngrams, 'loss': loss,
                                      'lr': lr, 'lr_update_rate': lr_update_rate, 'dim': dim,
                                      'bucket': bucket}).start()


def run_multi_classifiers():
    classifiers = []
    for file in os.listdir("./"):
        if file.endswith(".bin"):
            classifier = fasttext.load_model(file, label_prefix='__label__')
            classifiers.append(classifier)
    test_result(classifiers)


if __name__ == "__main__":
    train_classifiers(100)
