import os
import sys
from cv2 import cv2
import numpy as np
import pickle
import filteration as filter

if len(sys.argv) < 2:
    print("No file given.")
    exit()
if len(sys.argv) > 2:
    print("Too many inputs.")
    exit()

folder_name =  sys.argv[1]

sub_folders = os.listdir(folder_name)

letter_mapping = {}
count = 0
for sub in sub_folders:
    letter_mapping[sub] = count
    count += 1

training_data = np.zeros((45926,4096))
training_labels = np.zeros(45926)
testing_data = np.zeros((8121,4096))
testing_labels = np.zeros(8121)

for sub in sub_folders:
    all_images = os.listdir("./"+folder_name+"/"+sub)
    total_num = len(all_images)
    training_num = int(total_num*0.85)
    testing_num = total_num - training_num
    for i in range(total_num):
        img_path = "./"+folder_name+"/"+sub+"/"+all_images[i]
        print(sub, i)
        pre_image = cv2.imread(img_path, )
        image = filter.laplace(pre_image)
        flattened = image.flatten()/255.0
        if i < testing_num:
            print(len(flattened))
            if (len(flattened) > 4096):
                continue
            testing_data[i,:] = flattened
            testing_labels[i] = letter_mapping[sub]
        else:
            print(len(flattened))
            if (len(flattened) > 4096):
                continue
            training_data[i-testing_num,:] = flattened
            training_labels[i-testing_num] = letter_mapping[sub]
    
with open("../data/testing_data", 'wb') as f:
    pickle.dump(testing_data, f, pickle.HIGHEST_PROTOCOL)

with open("../data/testing_labels", 'wb') as f:
    pickle.dump(testing_labels, f, pickle.HIGHEST_PROTOCOL)

with open("../data/training_data", 'wb') as f:
    pickle.dump(training_data, f, pickle.HIGHEST_PROTOCOL)

with open("../data/training_labels", 'wb') as f:
    pickle.dump(training_labels, f, pickle.HIGHEST_PROTOCOL)
