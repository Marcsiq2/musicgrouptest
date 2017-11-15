import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import metrics

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
                        pass
                        #sample.append(d['sample_'+str(nsample)][feat][0])
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
    print 'Created training and testing sets with the following number of sounds:\n\tTrain\tTest\tTotal\tClass'
    for class_name in train:
        print '\t%i\t%i\t%i\t%s' % (len(train[class_name]), len(test[class_name]), len(train[class_name])+len(test[class_name]), class_name)

def computeGMMS(train, n_components):
    gmms = dict()
    for clas in train:
        features = train[clas]

        gmms[clas] = mixture.GaussianMixture(n_components)
        gmms[clas].fit(features)
    return gmms

def scoreGMMS(gmms, test):
    correct = []
    predicted = []
    for genre in test:
        for item in test[genre]:
            results = dict()
            x = np.array(item).reshape(1,-1)
            for g in gmms:
                results[g] = gmms[g].score(x)

            predicted.append(max(results, key=results.get))
            correct.append(genre)

    print "\nClassification report\n"
    print metrics.classification_report(correct, predicted)
    print "Confusion Matrix\n"
    print metrics.confusion_matrix(correct, predicted, test.keys())
    plotConfusionMatrix(test.keys(), correct, predicted)

def plotConfusionMatrix(labels, correct, predicted):
    cm = metrics.confusion_matrix(correct, predicted, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()