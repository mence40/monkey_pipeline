#All in one script. Will need to be partitioned for deployment.
#When compiling, use the commandline compiler and include dependencies
#import cv2
import os
import cv2
import random
import numpy as np
from scipy.linalg import sqrtm, inv
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def gen_target_transform(target_data):
    return sqrtm(np.cov(target_data, rowvar=False) + np.eye(target_data.shape[1]))

def decorrelate(data):
    return np.matmul(data, inv(sqrtm(np.cov(data, rowvar=False) + np.eye(data.shape[1]))))

def load(dir, target_label, vocab, include_target=False, transfer_dom=False, trg_transform=None):
    data = []
    labels=[]
    for label in os.listdir(dir):
        classification = 0
        if label == target_label:
            classification = 1
            if not include_target:
                continue
        descrip_data = []
        path = dir + '/' + label
        for name in os.listdir(path):
            image=cv2.imread(path + '/' + name)
            keyp=orb.detect(image, None)
            key, des = orb.compute(image, keyp)
            descrip_data.append(des)
        features = np.zeros((len(descrip_data), 800), "float32")
        for i in range(len(descrip_data)):
            words,dist = vq(descrip_data[i], vocab)
            for w in words:
                features[i][w]+=1
        scaler = StandardScaler().fit(features)
        features=scaler.transform(features)
        if transfer_dom:
            transformed_data = np.matmul(decorrelate(features), trg_transform)
            data = np.vstack((data, transformed_data)) if len(data) else transformed_data
            labels += [classification] * len(features)
        else:
            data = np.vstack((data, features)) if len(data) else features
            labels += [classification] * len(features)
    return data, labels

training_path = 'C:/Users/Sam/Documents/monkey_pipeline/data/training/training'
valid_path =  'C:/Users/Sam/Documents/monkey_pipeline/data/validation/validation'


#Requires two arguments to work: label for the monkey species and k for k-shot learning
target_species = 'n0'
k = 5
cross_validation_splits = 5

#
train_files = []
for image in os.listdir(training_path + "/" + target_species):
    imgpath = training_path + "/" + target_species + "/" + image
    train_files.append(imgpath)

random.shuffle(train_files)
train_files = train_files[:k]

orb = cv2.ORB_create()
descrip = []
for path in train_files:
    image=cv2.imread(path)
    keyp=orb.detect(image, None)
    key, des = orb.compute(image, keyp)
    descrip.append((path, des))

descriptors = descrip[0][1]
for path, desc in descrip[1:]:
    descriptors = np.vstack((descriptors, desc))
descriptors_f = descriptors.astype(float)

words=800
vocab, var = kmeans(descriptors_f, words, 1)
features = np.zeros((k, words), "float32")
for i in range(k):
    words,dist = vq(descrip[i][1], vocab)
    for w in words:
        features[i][w]+=1
scaler = StandardScaler().fit(features)
features=scaler.transform(features)

#important
trg_transform = gen_target_transform(features).astype(float)

train_labels = [1] * len(features)
train_data = features

#An 800 word vocabulary has been generated to represent the images. This vocabulary needs to be applied to the transofrmed data.
#This should probably be the end of the first function.
# Requirements of next part: target class name, vocab, recorrelation matrix 

new_train_data, new_labels = load(training_path, target_species, vocab, include_target=False, transfer_dom=True, trg_transform=trg_transform)
train_labels += new_labels
train_labels = np.array(train_labels)
train_data=np.vstack((train_data, new_train_data))
train_data = train_data.astype(float)

#all data should now be transformed into the same domain. Last step is to load the test set and train a classifier
valid_data, valid_labels = load(valid_path, target_species, vocab, include_target=True, transfer_dom=False)
classifier = LinearSVC(max_iter=80000)
classifier.fit(train_data, train_labels)
valid_predictions = classifier.predict(valid_data)
print('RESULTS:, valid_acc=' + str(accuracy_score(valid_predictions, valid_labels)))

print('done')