import re
import time
from collections import defaultdict
from random import choice

import fasttext
from numpy import argsort
from scipy import stats
from sklearn import cluster
from udicOpenData.stopwords import *


def random_string(length):
    return ''.join(choice('0123456789ABCDEF') for i in range(length))


def convert_to_pair(lines):
    pair_dict = defaultdict(str)
    for line in lines:
        if len(re.split(r"__label__\w+ ", line)) == 2 and len(re.findall(r"__label__\w+ ", line)) > 0:
            label = re.findall(r"__label__\w+ ", line)[0].strip().split("__label__")[1]
            sentence = re.split(r"__label__\w+ ", line)[1].strip()
            pair_dict[sentence] = label
    return pair_dict


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
def train_classifier(save_dri="./", epoch=300, word_ngrams=7, loss='hs', lr=0.01, lr_update_rate=10, dim=300,
                     bucket=200000, training_loc=''):
    save_loc = save_dri + 'lr_' + str(lr) + "_" + str(int(time.time())) + random_string(2)
    classifier = fasttext.supervised(training_loc,
                                     save_loc,
                                     label_prefix='__label__', silent=False,
                                     lr=lr, lr_update_rate=lr_update_rate, epoch=epoch, dim=dim, bucket=bucket,
                                     word_ngrams=word_ngrams, loss=loss)
    return classifier


def test_classifier(classifier, testing_file):
    result = classifier.test(testing_file)
    print(result.precision)
    print(result.recall)
    return result


def train_vector(input='./training_data/taipei/Taipei_QA_plain.txt', output='model', lr=0.1, dim=300):
    fasttext.skipgram(input, output, lr, dim)


def load_model(target_dir, cb_queue):
    cb_queue.put(fasttext.load_model(target_dir, label_prefix='__label__'))


def sentence2vec(model, sentence):
    return model[''.join(rmsw(sentence, flag=False))]


def predict_combine_with_classifiers(classifiers, input_sentence, true_label=''):
    output_dict = defaultdict(str)
    classifier_output_prob = defaultdict(list)
    mixed_output_result = defaultdict(int)
    prob_denominator = 0
    for classifier in classifiers:
        prob = classifier.predict_proba([input_sentence])[0][0]
        (classifier_output_prob[prob[0]]).append(prob[1])
        prob_denominator += prob[1]
    for key, value in classifier_output_prob.items():
        mixed_output_result[key] = sum(value) / prob_denominator
    max_label = max(mixed_output_result, key=mixed_output_result.get)
    output_dict["true_label"] = true_label
    output_dict["entropy"] = stats.entropy(list(mixed_output_result.values()))
    output_dict["max_label"] = max(mixed_output_result, key=mixed_output_result.get)
    output_dict["max_label_prob"] = mixed_output_result[max(mixed_output_result, key=mixed_output_result.get)]
    output_dict["correct"] = true_label == max_label
    return output_dict


def predict_with_each_classifiers(classifiers, classifiers_loc, testing_loc='./training_data/taipei/testing.data'):
    detail_list_dict = defaultdict(dict)
    for i in range(len(classifiers)):
        detail_dict = defaultdict(str)
        classifier = classifiers[i]
        classifier_loc = classifiers_loc[i]
        result = classifier.test(testing_loc)
        detail_dict["precision"] = str(result.precision)
        detail_dict["recall"] = str(result.recall)
        detail_dict["dim"] = str(classifier.dim)
        detail_dict["ws"] = str(classifier.ws)
        detail_dict["epoch"] = str(classifier.epoch)
        detail_dict["neg"] = str(classifier.neg)
        detail_dict["word_ngrams"] = str(classifier.word_ngrams)
        detail_dict["loss_name"] = str(classifier.loss_name)
        detail_dict["bucket"] = str(classifier.bucket)
        detail_dict["minn"] = str(classifier.minn)
        detail_dict["maxn"] = str(classifier.maxn)
        detail_dict["lr"] = re.findall(r"lr_\d.\d+", classifier_loc)[0].replace("lr_", "")
        detail_dict["lr_update_rate"] = str(classifier.lr_update_rate)
        detail_dict["t"] = str(classifier.t)
        detail_list_dict[classifier_loc] = detail_dict
    return detail_list_dict


def get_kmeans_center(input_arr, clusters=50):
    infer_vectors = []
    print("load word vec model...")
    model_w2v = fasttext.load_model('model.bin')
    print("turn sentence to vector...")
    for sentence in input_arr:
        vector = sentence2vec(model_w2v, sentence)
        infer_vectors.append(vector)
    print("train kmeans model...")
    kmeans_model = cluster.KMeans(n_clusters=clusters)
    kmeans_model.fit(infer_vectors)
    kmeans_model.predict(infer_vectors)
    print("kmeans predict")
    result_dict = {}
    for i in range(clusters):
        d = kmeans_model.transform(infer_vectors)[:, i]
        ind = argsort(d)[::][:1]
        result_dict.setdefault(i, input_arr[ind[0]])
    print("out put result...")
    result_list = []
    for kcluster, center_sentence in result_dict.items():
        result_list.append(center_sentence)

    return result_list
