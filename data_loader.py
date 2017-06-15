import numpy as np
import csv

class DataLoader(object):

    def __init__(self, flags, data_processor):
        self.__flags = flags
        self.__data_processor = data_processor
        self.__train_data_file = None
        self.__dev_data_file = None
        self.__class_data_file = None
        self.__classes_cache = None

    def define_flags(self):
        self.__flags.DEFINE_string("train_data_file", "data-sms_word_vector.csv", "Data source for the training data.")
        self.__flags.DEFINE_string("dev_data_file", "test-sms_word_vector.csv", "Data source for the test data")
        self.__flags.DEFINE_string("bag_of_word_data_file", "bow.out", "Data source for the Bag of Words")

    def prepare_data(self):
        self.__resolve_params()
        x_train, y_train = self.__load_data_and_labels(self.__train_data_file)
        x_dev, y_dev = self.__load_data_and_labels(self.__dev_data_file)
        return [x_train, y_train, x_dev, y_dev]


    def __load_data_and_labels(self, data_file):
        x = []
        y = []

        with open(data_file, 'r') as csvin :
            one_hot_vectors = np.eye(2, dtype=int)
            csvin = csv.reader(csvin, delimiter=',')
            for row in csvin:
                # data = self.__data_processor.clean_data(row[0])
                x.append(np.array(row[1:]))
                y.append(one_hot_vectors[int(row[0])])

        return [np.array(x), np.array(y)]


    def __bow(self):
        self.__resolve_params()
        if self.__classes_cache is None:
            classes = {}
            with open(self.__class_data_file, 'r') as csvin:
                csvin = csv.reader(csvin, delimiter=",")
                for row in csvin:
                    classes[row[0]] = row[2]
            self.__classes_cache = classes
        return self.__classes_cache


    def get_word_by_id(self, id):
        return self.__bow()[id]

    def get_bow_size(self):
        return len(self.__bow().keys())
    
    def get_bow_words(self):
        return self.__bow().keys()


    def __resolve_params(self):
        if self.__class_data_file is None:
            self.__train_data_file = self.__flags.FLAGS.train_data_file
            self.__dev_data_file = self.__flags.FLAGS.dev_data_file
            self.__class_data_file = self.__flags.FLAGS.bag_of_word_data_file