import os
import pickle 

def load_obj(filename):
    load = {}
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            load = pickle.load(f)
    return load



def get_data(testing_data_path, testing_labels_path,training_data_path, training_labels_path):
    testing_data = load_obj(testing_data_path)
    testing_labels = load_obj(testing_labels_path)
    training_data = load_obj(training_data_path)
    training_labels = load_obj(training_labels_path)
    print(len(testing_data), len(testing_labels), len(training_data), len(training_labels))
    return testing_data, testing_labels, training_data, training_labels


get_data("../data/testing_data", "../data/testing_labels","../data/training_data", "../data/training_labels")

# print(len(testing_data), len(testing_labels), len(training_data), len(training_labels))
