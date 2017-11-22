import os
import json
import random
import numpy as np
from sklearn import mixture
from sklearn import metrics
from sklearn import preprocessing

def fetchFiles(inputDir, descExt):
    files = []
    for path, dname, fnames  in os.walk(inputDir):
        for fname in fnames:
            if descExt in fname.lower():
                files.append((path, fname))
    return files

def importData(files):
    database = dict()
    for path, fname in files:
        clas = os.path.basename(os.path.normpath(path))
        if clas not in database:
            database[clas] = []
        with open(path + "/"+ fname) as json_data:
            d = json.load(json_data)
            for nsample in range(1,d['numSamples'][0]+1):
                sample = []
                for feat in d['featTypes'][0]:
                    if feat in ['mfcc', 'gfcc', 'lpc']:
                        sample.extend(d['sample_'+str(nsample)][feat][0])
                    else:
                        sample.append(d['sample_'+str(nsample)][feat][0])
                database[clas].append(sample)
    return database

def randomTrainTest(data, percentage_train):
    train = dict()
    test = dict()

    for class_name, items in data.items():
        items_class = items[:]
        train_per_class = int(np.ceil(len(items_class) * percentage_train))
        random.shuffle(items_class)
        train[class_name] = items_class[:train_per_class]
        test[class_name] = items_class[train_per_class:]

    return train, test

def printDataStats(train, test):
    print 'Created training and testing sets with the following number of samples:\n\n\tTrain\tTest\tTotal\tClass\n'
    tr = 0
    te = 0
    for class_name in train:
        print '\t%i\t%i\t%i\t%s' % (len(train[class_name]), len(test[class_name]), len(train[class_name])+len(test[class_name]), class_name)
        tr = tr+len(train[class_name])
        te = te+len(test[class_name])
    print '\n\t%i\t%i\t%i\t%s' % (tr, te, tr+te, 'TOTAL')

def computeGMMS(data, n_components, covariance_type='full'):
    gmms = dict()
    for clas in data:
        gmms[clas] = mixture.GaussianMixture(n_components, covariance_type)
        gmms[clas].fit(data[clas])
    return gmms

def scoreGMMS(gmms, test):
    correct = []
    predicted = []
    for clas in test:
        for item in test[clas]:
            predicted.append(predict(gmms, item))
            correct.append(clas)

    return correct, predicted

def predict(gmms, sample):
    results = dict()
    x = np.array(sample).reshape(1,-1)
    for g in gmms:
        results[g] = gmms[g].score(x)
    return max(results, key=results.get)

def convertToXY(data):
    X = []
    Y = []
    for clas in data:
        for item in data[clas]:
            X.append(item)
            Y.append(clas)
    return X, Y

def normalize_data(data):
    X, Y = convertToXY(data)

    normalized = []
    for idx in range(len(X[0])):
        feat = np.array([item[idx] for item in X])
        prec = preprocessing.normalize(feat.reshape(1, -1))
        normalized.append(prec)

    id = 0
    data_norm = dict()

    for clas in data:
        if not clas in data_norm:
            data_norm[clas] = []
        for idx, pool in enumerate(data[clas]):
            x_norm = []
            for idx2 in range(0, len(normalized)):
                mf = [item[id] for item in normalized[idx2]][0]
                x_norm.append(mf)

            data_norm[clas].append(x_norm)
            id += 1

    return data_norm


def classificationReport(correct, predicted):
    print "\nClassification report\n"
    print metrics.classification_report(correct, predicted)
    print "Confusion Matrix\n"
    print metrics.confusion_matrix(correct, predicted)

def runXtimes(data, percentage_train=0.3, n_components=2, covariance_type='full', nruns=10, printa=False):
    stats = []
    if printa:
        print 'Iter\tPrec\tRec\tf1-s\n'
    for n in range(nruns):
        train, test = randomTrainTest(data, percentage_train)
        gmms = computeGMMS(train, n_components)
        correct, predicted = scoreGMMS(gmms, test)
        precision, recall, f1score, support = metrics.precision_recall_fscore_support(correct, predicted, average='weighted')
        stats.append([precision, recall, f1score])
        if printa:
            print '%i\t%.3f\t%.3f\t%.3f' % (n+1,precision, recall, f1score)
    if printa:
        print '\navg:\t' + '\t'.join(['%.3f' % a for a in np.average(stats, axis=0)]) + '\n'
    return np.average(stats, axis=0)
