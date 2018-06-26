import queue
import threading
import time
from collections import defaultdict
from multiprocessing import Process
from random import uniform, randint, sample

from nlp2 import *
from numpy import *

import fastText_utility


def run_multi_process(target, args_list):
    q = queue.Queue()
    result = []
    threading_pool = []
    for args in args_list:
        threading_pool.append(threading.Thread(target=target, args=(args, q)))
    for i in threading_pool:
        i.start()
    for i in threading_pool:
        i.join()
    for _ in threading_pool:
        result.append(q.get())
    return result


def train_multi_classifier(training_loc='./', model_num=10, save_dri="./" + str(int(time.time())) + "/"):
    prepare_train_dir(save_dri)
    loss_method = ['ns', 'hs', 'softmax']
    p_list = []
    for i in range(model_num):
        epoch = randint(5, 300)
        word_ngrams = randint(1, 12)
        lr = uniform(0.01, 0.99)
        lr_update_rate = randint(5, 500)
        loss = loss_method[randint(0, 2)]
        dim = randint(100, 700)
        bucket = randint(5000, 20000)
        p = Process(target=fastText_utility.train_classifier,
                    kwargs={'save_dri': save_dri, 'epoch': epoch, 'word_ngrams': word_ngrams, 'loss': loss,
                            'lr': lr, 'lr_update_rate': lr_update_rate, 'dim': dim,
                            'bucket': bucket, 'training_loc': training_loc})
        p_list.append(p)
        p.start()
    for i in p_list:
        i.join()
    return save_dri


def get_multi_classifier_loc(classifier_dir="./"):
    classifiers_loc = []
    for path in get_files_from_dir(classifier_dir):
        classifiers_loc.append(path)
    return classifiers_loc


def filter_loc_classifiers(threshold=0.5, classifier_dir="./", testing_data=''):
    classifiers_loc = get_multi_classifier_loc(classifier_dir)
    classifiers = run_multi_process(fastText_utility.load_model, classifiers_loc)
    detail = fastText_utility.predict_with_each_classifiers(classifiers, classifiers_loc, testing_data)
    for classifier_loc, value in detail.items():
        print(value["precision"])
        if float(value["precision"]) < threshold or float(value["recall"]) < threshold:
            os.remove(classifier_loc)


class Assesser:
    classifiers = []

    def __init__(self, classifier_dir):
        classifiers_loc = get_multi_classifier_loc(classifier_dir)
        self.classifiers = run_multi_process(fastText_utility.load_model, classifiers_loc)

    def evaluate(self, queries):
        if isinstance(queries, str): queries = [queries]
        result_dict = defaultdict(str)
        for querie in queries:
            sentence = ' '.join(querie)
            print(sentence)
            output_detail = fastText_utility.predict_combine_with_classifiers(self.classifiers, sentence)
            result_dict[sentence] = output_detail
            print(sentence + " : " + str(output_detail['entropy']))
        return result_dict


def test_multi_classifier_bagging(classifier_dir="./", testing_loc=''):
    classifiers_loc = get_multi_classifier_loc(classifier_dir)
    classifiers = run_multi_process(fastText_utility.load_model, classifiers_loc)
    mixed_classifier_result = defaultdict(dict)
    with open(testing_loc, "r", encoding='utf-8') as f:
        testing_data = fastText_utility.convert_to_pair(f.readlines())
    true_count = 0
    print(len(testing_data))
    for sentence, label in testing_data.items():
        output_detail = fastText_utility.predict_combine_with_classifiers(classifiers, sentence, label)
        mixed_classifier_result[sentence] = output_detail
        true_count += output_detail["correct"]
    print(classifier_dir + " : accuracy = " + ("%.3f" % (true_count / len(testing_data))))
    return mixed_classifier_result, str("%.3f" % (true_count / len(testing_data)))


def store_bagging_detail(mixed_classifier_result, accuracy_score, output_name='bagging_detail'):
    with open(output_name, 'w', encoding='utf-8') as f:
        f.write("Accuracy : " + accuracy_score + "\n")
        for sent, detail in mixed_classifier_result.items():
            f.write(sent + "\n")
            for detail_key, value in detail.items():
                f.write(detail_key + " : " + str(value) + "\n")
            f.write("=====" + "\n")


def store_classifiers_detail(classifier_dir="./", testing_loc='', output_name='detail'):
    with open(output_name, 'w', encoding='utf-8') as f:
        classifiers_loc = get_multi_classifier_loc(classifier_dir)
        classifiers = run_multi_process(fastText_utility.load_model, classifiers_loc)
        result = fastText_utility.predict_with_each_classifiers(classifiers, classifiers_loc, testing_loc)
        for loc, detail in result.items():
            f.write(loc + "\n")
            for sent_key, value in detail.items():
                f.write(sent_key + " : " + value + "\n")
            f.write("=====" + "\n")


def select_retain_data_random(training_loc, testing_loc, output_loc, select_count=50):
    with open(training_loc, 'r', encoding='utf-8') as training:
        training_lines = training.readlines()
    with open(testing_loc, 'r', encoding='utf-8') as f:
        testing_pairs = fastText_utility.convert_to_pair(f.readlines())
    with open(output_loc + "training_random.data", 'w', encoding='utf-8') as output_training:
        random_list = sample(list(testing_pairs.keys()), select_count)
        for i in training_lines:
            output_training.write(i)
        for i in random_list:
            output_training.write("__label__" + testing_pairs[i] + " " + i + "\n")
            testing_pairs.pop(i, None)
    with open(output_loc + "testing_random.data", 'w', encoding='utf-8') as output_testing:
        for sent, label in testing_pairs.items():
            output_testing.write("__label__" + label + " " + sent + "\n")


def select_retain_data_entropy(training_loc, bagging_res, output_loc, select_count=50):
    with open(training_loc, 'r', encoding='utf-8') as training:
        training_lines = training.readlines()
    with open(output_loc + "training_entropy.data", 'w', encoding='utf-8') as output_training:
        mixed_classifier_result = sorted(bagging_res.keys(),
                                         key=lambda t: bagging_res[t]['entropy'], reverse=True)
        for i in training_lines:
            output_training.write(i)
        for i in range(select_count):
            sent = mixed_classifier_result[i]
            output_training.write("__label__" + bagging_res[sent]["true_label"] + " " + sent + "\n")
    with open(output_loc + "testing_entropy.data", 'w', encoding='utf-8') as output_testing:
        for i in range(select_count, len(mixed_classifier_result)):
            sent = mixed_classifier_result[i]
            output_testing.write("__label__" + bagging_res[sent]["true_label"] + " " + sent + "\n")


def select_retain_data_kmeans(training_loc, testing_loc, output_loc, select_count=50):
    with open(training_loc, 'r', encoding='utf-8') as training:
        training_lines = training.readlines()
    with open(testing_loc, 'r', encoding='utf-8') as testing:
        testing_pairs = fastText_utility.convert_to_pair(testing.readlines())
    with open(output_loc + "training_kmeans.data", 'w', encoding='utf-8') as output_training:
        kmeans_list = fastText_utility.get_kmeans_center(list(testing_pairs.keys()), select_count)
        for i in training_lines:
            output_training.write(i)
        for i in kmeans_list:
            output_training.write("__label__" + testing_pairs[i] + " " + i + "\n")
            testing_pairs.pop(i, None)
    with open(output_loc + "testing_kmeans.data", 'w', encoding='utf-8') as output_testing:
        for sent, label in testing_pairs.items():
            output_testing.write("__label__" + label + " " + sent + "\n")


def prepare_train_dir(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    else:
        file_list = os.listdir(dirPath)
        for fileName in file_list:
            os.remove(dirPath + "/" + fileName)


def retrain_model(classifiers_dir='', output_dir='', training_data_dir='', testing_data_dir=''):
    prepare_train_dir(output_dir)
    classifiers_loc = get_multi_classifier_loc(classifiers_dir)
    classifiers = run_multi_process(fastText_utility.load_model, classifiers_loc)
    result = fastText_utility.predict_with_each_classifiers(classifiers, classifiers_loc, testing_data_dir)
    p_list = []
    for _, detail in result.items():
        epoch = detail['epoch']
        word_ngrams = detail['word_ngrams']
        lr = detail['lr']
        lr_update_rate = detail['lr_update_rate']
        loss = detail['loss_name']
        dim = detail['dim']
        bucket = detail['bucket']
        p = Process(target=fastText_utility.train_classifier,
                    kwargs={'save_dri': output_dir, 'epoch': epoch, 'word_ngrams': word_ngrams,
                            'loss': loss, 'lr': lr, 'lr_update_rate': lr_update_rate, 'dim': dim, 'bucket': bucket,
                            'training_loc': training_data_dir})
        p_list.append(p)
        p.start()
    for i in p_list:
        i.join()


if __name__ == "__main__":
    # fastText_utility.train_vector(input='./training_data/taipei/training.data', output='model', lr=0.1, dim=300)
    #
    # number_of_classifier = 100
    # filter_threshold = 0.5
    # origin_model_dir = train_multi_classifier('./training_data/taipei/training.data', number_of_classifier)
    # filter_loc_classifiers(filter_threshold, origin_model_dir, './training_data/taipei/training.data')

    origin_model_timestamp = "1529666650"
    origin_model_dir = "./" + origin_model_timestamp + "/"
    origin_training_dir = './training_data/taipei/training.data'
    origin_testing_dir = './training_data/taipei/testing.data'

    origin_bagging_result, origin_accuracy = test_multi_classifier_bagging(origin_model_dir,
                                                                           origin_testing_dir)
    store_bagging_detail(origin_bagging_result, origin_accuracy,"baseline.detail")

    # 10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 400, 500
    # 600, 700,800,900,1000,1100,1200,1300,1400,1500
    for exec_num in [10, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                     1300, 1400, 1500]:
        # retrain
        origin_bagging_result, origin_accuracy = test_multi_classifier_bagging(origin_model_dir,
                                                                               origin_testing_dir)
        select_retain_data_entropy(origin_training_dir, origin_bagging_result,
                                   "./training_data/taipei/", exec_num)
        select_retain_data_random(origin_training_dir, origin_testing_dir,
                                  "./training_data/taipei/", exec_num)
        select_retain_data_kmeans(origin_training_dir, origin_testing_dir,
                                  "./training_data/taipei/", exec_num)
        for ways in ["kmeans", "entropy", "random"]:
            retrain_model_dir = "./" + origin_model_timestamp + "_" + ways + "_" + str(exec_num) + "/"
            retrain_model(origin_model_dir, retrain_model_dir,
                          "./training_data/taipei/training_" + ways + ".data",
                          "./training_data/taipei/testing_" + ways + ".data")
            bagging_result, accuracy = test_multi_classifier_bagging(retrain_model_dir,
                                                                     "./training_data/taipei/testing_" + ways + ".data")
            store_bagging_detail(bagging_result, accuracy,
                                 ways + "_classifiers_bagging_" + str(exec_num) + ".detail")
            store_classifiers_detail(retrain_model_dir, "./training_data/taipei/testing_" + ways + ".data",
                                     ways + "_classifiers_" + str(exec_num) + ".detail")
