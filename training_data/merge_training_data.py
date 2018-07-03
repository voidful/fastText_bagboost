from nlp2 import *
from sklearn.model_selection import train_test_split

data_x = []
data_y = []
testing_ratio = 0.3

folder = './pixnet/'
origin_file_name = 'Pixnet_QA.txt'
if __name__ == '__main__':
    with open(folder + origin_file_name, "r", encoding='utf-8') as f:
        for i in f.readlines():
            label = re.split(" ", i)[0]
            question = re.split(" ", i)[1].rstrip()
            processed_question = ' '.join(spilt_sentence_to_array(question))
            data_x.append(processed_question)
            data_y.append(label)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.7)

    with open(folder + 'training.data', "w", encoding='utf-8') as f:
        for i in range(len(x_train)):
            f.write(y_train[i] + " " + x_train[i] + "\n")

    with open(folder + 'testing.data', "w", encoding='utf-8') as f:
        for i in range(len(x_test)):
            f.write(y_test[i] + " " + x_test[i] + "\n")
