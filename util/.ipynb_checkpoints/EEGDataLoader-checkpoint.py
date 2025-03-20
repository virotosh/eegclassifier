import scipy.io
import numpy as np
import torch

class EEGDataLoader:

    def __init__(self, data_dir, params):
        """For initialization"""
        self.params = params
        
        self.data_dir = data_dir

        self.trainData = None
        self.trainLabel = None
        self.testData = None
        self.testLabel = None

        # loading data
        self.load_data()

    def load_data(self):
        train_data = []
        train_label = []

        subjectID = self.params['subjectID']
        
        datatype = "T" # T is train, E is test/experimental
        for session_index in [1,2,3]:
            target_tmp = scipy.io.loadmat(self.data_dir + 'B0%d0%d%s.mat' % (subjectID, session_index,datatype))
            train_data_tmp = target_tmp['data']
            train_label_tmp = target_tmp['label']
            train_data_tmp = np.transpose(train_data_tmp, (2, 1, 0))
            train_data_tmp = np.expand_dims(train_data_tmp, axis=1)
            train_label_tmp = np.transpose(train_label_tmp)
            train_label_tmp = train_label_tmp[0]
            train_data.append(train_data_tmp)
            train_label.append(train_label_tmp)

        self.trainData = np.concatenate(train_data)
        self.trainLabel = np.concatenate(train_label)

        shuffle_num = np.random.permutation(len(self.trainData))
        self.trainData = self.trainData[shuffle_num, :, :, :]
        self.trainLabel = self.trainLabel[shuffle_num]
        self.trainLabel = self.trainLabel - 1 # class 1,2 to 0,1

        # test data
        test_data = []
        test_label = []
        datatype = "E" # T is train, E is test/experimental
        for session_index in [4,5]:
            test_tmp = scipy.io.loadmat(self.data_dir + 'B0%d0%d%s.mat' % (subjectID, session_index,datatype))
            test_data_tmp = test_tmp['data']
            test_label_tmp = test_tmp['label']
            test_data_tmp = np.transpose(test_data_tmp, (2, 1, 0))
            test_data_tmp = np.expand_dims(test_data_tmp, axis=1)
            test_label_tmp = np.transpose(test_label_tmp)
            test_label_tmp = test_label_tmp[0]
            test_data.append(test_data_tmp)
            test_label.append(test_label_tmp)
        self.testData = np.concatenate(test_data)
        self.testLabel = np.concatenate(test_label)
        self.testLabel = self.testLabel - 1 # class 1,2 to 0,1

        # normalize
        target_mean = np.mean(self.trainData)
        target_std = np.std(self.trainData)
        self.trainData = (self.trainData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

    