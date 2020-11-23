import os
import pickle 
import tensorflow as tf
import numpy as np

def load_obj(filename):
    load = {}
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            load = pickle.load(f)
    return load



def get_data(testing_data_path, testing_labels_path,training_data_path, training_labels_path):
    testing_data = tf.transpose(tf.reshape(load_obj(testing_data_path),(-1,1,64,64)),perm=[0,2,3,1])
    testing_labels = tf.one_hot(load_obj(testing_labels_path),32,dtype=tf.float32)
    training_data = tf.transpose(tf.reshape(load_obj(training_data_path),(-1,1,64,64)),perm=[0,2,3,1])
    training_labels = tf.one_hot(load_obj(training_labels_path),32,dtype=tf.float32)
    print(testing_data.shape, testing_labels.shape, training_data.shape, training_labels.shape)
    return testing_data, testing_labels, training_data, training_labels


get_data("../data/testing_data", "../data/testing_labels","../data/training_data", "../data/training_labels")
