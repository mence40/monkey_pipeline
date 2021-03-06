# All in one script. Will need to be partitioned for deployment.
# When compiling, use the commandline compiler and include dependencies
# import cv2
import json
import os
import random
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import boto3
import logging


def lambda_handler(event=None, context=None):
    def gen_target_transform(target_data):
        return np.sqrt(np.cov(target_data, rowvar=False) + np.eye(target_data.shape[1]))

    def decorrelate(data):
        return np.matmul(data, np.linalg.inv(np.sqrt(np.cov(data, rowvar=False) + np.eye(data.shape[1]))))

    def load(dir, target_label, vocab, include_target=False, transfer_dom=False, trg_transform=None):
        data = []
        labels = []
        for label in os.listdir(dir):
            classification = 0
            if label == target_label:
                classification = 1
                if not include_target:
                    continue
            descrip_data = []
            path = dir + '/' + label
            for name in os.listdir(path):
                descrip_data.append(np.load(dir + '/' + label + '/' + name))
            features = np.zeros((len(descrip_data), 800), "float32")
            for i in range(len(descrip_data)):
                words, dist = vq(descrip_data[i], vocab)
                for w in words:
                    features[i][w] += 1
            scaler = StandardScaler().fit(features)
            features = scaler.transform(features)
            if transfer_dom:
                transformed_data = np.matmul(decorrelate(features), trg_transform).astype(float)
                data = np.vstack((data, transformed_data)) if len(data) else transformed_data
                labels += [classification] * len(features)
            else:
                data = np.vstack((data, features)) if len(data) else features
                labels += [classification] * len(features)
        return data, labels

    training_path = '/tmp/training'
    valid_path = '/tmp/validation'

    # Requires two arguments to work: label for the monkey species and k for k-shot learning
    target_species = 'n0'
    k = 5
    cross_validation_splits = 5

    bucketName = "monkey-pipeline--comp-a--raw-data"

    #bucket = s3_resource.Bucket(bucketName)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    client = boto3.client('s3')
    resource = boto3.resource('s3')

    download_dir(client, resource, "training/", '/tmp', bucket=bucketName)
    download_dir(client, resource, "validation/", '/tmp', bucket=bucketName)

    # for root, directories, filenames in os.walk('/tmp/'):
    #     logging.info("made it in")
    #     for directory in directories:
    #         logging.info(os.path.join(root, directory))
    #
    #     for filename in filenames:
    #         logging.info(os.path.join(root, filename))

    #FUNCTION 1
    train_files = []
    for image in os.listdir(training_path + "/" + target_species):
        imgpath = training_path + "/" + target_species + "/" + image
        train_files.append(imgpath)

    random.shuffle(train_files)
    train_files = train_files[:k]

    #orb = cv2.ORB_create()
    descrip = []
    for path in train_files:
        descrip.append(np.load(path))

    # Fixed
    #logging.info(descrip[1:])

    descriptors = descrip[0]
    for desc in descrip[1:]:
        descriptors = np.vstack((descriptors, desc))
    descriptors_f = descriptors.astype(float)

    words = 800
    vocab, var = kmeans(descriptors_f, words, 1)
    features = np.zeros((k, words), "float32")
    for i in range(k):
        # Fiexed
        # logging.info(vq(descrip[i], vocab))
        words, dist = vq(descrip[i], vocab)
        for w in words:
            features[i][w] += 1
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    # important
    trg_transform = gen_target_transform(features).astype(float)

    #FUNCTION 2
    #WHEN SEPARATED FROM 1, THESE NEED TO BE PASSED:
    #1. TRG_TRANSFORM, JUST MAKE A NP DUMP LIKE "target_class_transform.npy" AND LOAD
    #2. TARGET CLASS

    train_labels = [1] * len(features)
    train_data = features

    assert (np.isfinite(train_data).all())

    # An 800 word vocabulary has been generated to represent the images. This vocabulary needs to be applied to the transofrmed data.
    # This should probably be the end of the first function.
    # Requirements of next part: target class name, vocab, recorrelation matrix

    new_train_data, new_labels = load(training_path, target_species, vocab, include_target=False, transfer_dom=True,
                                      trg_transform=trg_transform)
    train_labels += new_labels
    train_labels = np.array(train_labels)
    train_data = np.vstack((train_data, new_train_data))

    assert (np.isfinite(train_data).all())

    for line in train_data:
        logging.info(line)

    train_data = train_data.astype(float)

    assert (np.isfinite(train_data).all())

    #FUNCTION 3
    #WHEN SEPARATED FROM 2, THESE NEED TO BE PASSED:
    #1. TARGET CLASS
    #2. TRAIN_DATA, SHOULD BE SAVED AS NP DUMP IN 2
    #3. TRAIN_LABELS

    # all data should now be transformed into the same domain. Last step is to load the test set and train a classifier
    valid_data, valid_labels = load(valid_path, target_species, vocab, include_target=True, transfer_dom=False)
    classifier = LinearSVC(max_iter=80000)
    logging.info(train_data)
    logging.info(train_labels)
    classifier.fit(train_data, train_labels)
    valid_predictions = classifier.predict(valid_data)
    print('RESULTS:, valid_acc=' + str(accuracy_score(valid_predictions, valid_labels)))
    logging.info('RESULTS:, valid_acc=' + str(accuracy_score(valid_predictions, valid_labels)))
    print('done')

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):

    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    #
    paginator = client.get_paginator('list_objects')
    #
    # logging.info(paginator)
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):

        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)
