from utils import *

databaseDir = 'D:\Users\Marc\Downloads\music_group_ml_test\music_group_ml_test\music_group_ml_test_data'
jsonfiles = fetchFiles(databaseDir, '.json')
print "Number of json files fetched: " + str(len(jsonfiles))

data = importData(jsonfiles)

train, test = randomTrainTest(data, 0.3)

printDataStats(train, test)

gmms = computeGMMS(train, 1)

scoreGMMS(gmms, test)
