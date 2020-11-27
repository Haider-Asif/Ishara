from __future__ import absolute_import
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_array_ops import pad
import tensorflow as tf
from load import get_data

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. Do not modify the constructor, as doing so 
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 32
        self.training_loss_list = []
        self.test_loss_list = []
        # TODO: Initialize all hyperparameters
        self.hidden_layer = 96
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # TODO: Initialize all trainable parameters
        ##filters
        self.conv_layer_1 = tf.keras.layers.Conv2D(32,3,strides=(2,2),padding="SAME",activation="relu")
        self.max_pool_1 = tf.keras.layers.MaxPool2D(3,strides=(2,2),padding="SAME")
        self.conv_layer_2 = tf.keras.layers.Conv2D(64,3,strides=(2,2),padding="SAME",activation="relu")
        self.max_pool_2 = tf.keras.layers.MaxPool2D(3,strides=(2,2),padding="SAME")
        self.conv_layer_3 = tf.keras.layers.Conv2D(self.hidden_layer,3,strides=(2,2),padding="SAME",activation="relu")
        self.max_pool_3 = tf.keras.layers.MaxPool2D(3,strides=(2,2),padding="SAME")
        self.flatten = tf.keras.layers.Flatten()
        self.Dense_1 = tf.keras.layers.Dense(self.hidden_layer,activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.Dense_2 = tf.keras.layers.Dense(self.num_classes)


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        # print("inputs shape", inputs.shape)
        out_1 = self.conv_layer_1(inputs)
        # print("shapes after conv 1", out_1.shape)
        out_2 = self.conv_layer_2(self.max_pool_1(out_1))
        # print("shapes after conv 2", out_2.shape)
        out_3 = self.max_pool_3(self.conv_layer_3(self.max_pool_2(out_2)))
        # print("shapes after conv 3", out_3.shape)
        flattened = self.flatten(out_3)
        # print("shapes after flattened", flattened.shape)
        dense_1 = self.dropout(self.Dense_1(flattened))
        # print("shapes after dense_1", dense_1.shape)
        return self.Dense_2(dense_1)

    def loss(self, logits, labels):

        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    indices = tf.random.shuffle([x for x in range(len(train_inputs))])
    train_inputs = tf.gather(train_inputs,indices)
    train_labels = tf.gather(train_labels,indices)
    for i in range(0,len(train_inputs),model.batch_size):
        batched_inputs = train_inputs[i:i+model.batch_size]
        batched_inputs = tf.image.random_flip_left_right(batched_inputs)
        batched_labels = train_labels[i:i+model.batch_size]
        with tf.GradientTape() as tape:
            predictions = model.call(batched_inputs)
            loss = model.loss(predictions,batched_labels)
            model.training_loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    acc = 0
    count = 0
    for i in range(0,len(test_inputs),model.batch_size):
        batched_inputs = test_inputs[i:i+model.batch_size]
        batched_labels = test_labels[i:i+model.batch_size]
        predictions = model.call(batched_inputs)
        loss = model.loss(predictions,batched_labels)
        model.test_loss_list.append(loss)
        # if count==5 or count==50 or count==100:
        #     for x in range(model.batch_size):
        print("predictions", tf.argmax(predictions,1), "labels", tf.argmax(batched_labels,1))
        acc += model.accuracy(predictions,batched_labels)
        count+=1
    return acc/(len(test_inputs)/model.batch_size)


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = predicted_labels[ind]
            al = np.argmax(image_labels[ind], axis=0)
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    test_inputs, test_labels, train_inputs, train_labels = get_data("../data/testing_data", "../data/testing_labels","../data/training_data", "../data/training_labels")
    num_epochs = 1
    model = Model()
    for i in range(num_epochs):
        print("EPOCH -", i)
        train(model,train_inputs,train_labels)
    accuracy = test(model,test_inputs,test_labels)
    print("Accuracy", accuracy)
    visualize_loss(model.training_loss_list)
    visualize_loss(model.test_loss_list)
    X,Y = train_inputs[0:10],train_labels[0:32]
    # visualize_results(X,model.call(X),Y)
    return


if __name__ == '__main__':
    main()
