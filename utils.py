import os
import csv
import json
import random
import itertools
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
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

    for class_name in train:
        print '\t%i\t%i\t%i\t%s' % (len(train[class_name]), len(test[class_name]), len(train[class_name])+len(test[class_name]), class_name)

    tr = sum(len(train[v]) for v in train)
    te = sum(len(test[v]) for v in test)
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


def analyzeGMMs(data, max_components=10, nruns=10, results_file='results.csv'):
    cv_types = ['spherical', 'tied', 'diag', 'full']
    color_iter = itertools.cycle(['c','y', 'r','b'])
    n_components_range = range(1, max_components+1)
    f1s = []
    bestf1s = 0

    with open(results_file, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['covariance_type', 'n_components', 'iteration', 'precision', 'recall', 'f1-score'])
        for cv_type in cv_types:
            for n_components in n_components_range:
                stats = []
                for n in range(1,nruns+1):
                    train, test = randomTrainTest(data, percentage_train=0.7)
                    gmms = computeGMMS(train, n_components, cv_type)
                    correct, predicted = scoreGMMS(gmms, test)
                    precision, recall, f1score, support = metrics.precision_recall_fscore_support(correct, predicted, average='weighted')
                    stats.append([precision, recall, f1score])
                    writer.writerow([cv_type, n_components, n, precision, recall, f1score])
                csvfile.flush()
                f1sc = np.average(stats, axis=0)
                f1s.append(f1sc[2])
                if f1s[-1] > bestf1s:
                    bestf1s = f1s[-1]
                    best_gmm = (cv_type, n_components)

    print '\n***** Best F1-score of {} --> with cov: {} and n_comp: {} *****'.format(bestf1s, best_gmm[0], best_gmm[1])

    #plotting the results
    f1s = np.array(f1s)
    bars = []
    
    fig=plt.figure(figsize=(12, 8))
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, f1s[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([0, 1])
    plt.title('F1-score per model')
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)
    plt.show()

    return best_gmm