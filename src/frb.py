from sklearn import datasets, cluster, preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import argparse, os

class FuzzyRuleBase:

    def __init__(self, data, target,numofclass,numofrulesperclass):
        self.nc = numofclass
        self.nrpc = numofrulesperclass
        self.nr = numofclass*numofrulesperclass
        self.dim = data.shape[-1]
        self.X = (map(lambda i: data[target == i],xrange(self.nc)))
        self.rule_target = np.arange( self.nr ).reshape((self.nc,self.nrpc))
        Y = (map(self.getcluster, xrange(self.nc)))
        M = np.asarray(map(lambda i,j:self.X[i][Y[i]==j].mean(axis=0),FuzzyRuleBase.myrange(self.nc,self.nrpc ), range(self.nrpc )*self.nc))
        S = np.asarray(map(lambda i,j:self.X[i][Y[i]==j].std(axis=0),FuzzyRuleBase.myrange(self.nc,self.nrpc ), range(self.nrpc )*self.nc))
        self.m = M
        self.s = S
        self.s2 = np.square(S)

    def getcluster(self,k):
        k_means = cluster.KMeans(self.nrpc)
        k_means.fit(self.X[k])
        return k_means.labels_
    
    def gaussianMF(self,datapoint):
        mfv = np.subtract(datapoint,self.m)
        np.square(mfv,mfv)
        np.negative(mfv,mfv)
        np.divide(mfv,self.s2,mfv)
        np.exp(mfv,mfv)
        #mfv = np.exp(np.divide(np.negative(np.square(np.subtract(datapoint,model.m))),np.square(model.s))
        #inline version maybe slower
        return mfv               
    
    def getFS(self,datapoint):
        fss = np.min(self.gaussianMF(datapoint), axis=1)
        fss = np.asarray(map(lambda x:(fss[self.rule_target[x]]).max(),xrange(self.nc)))
        return fss
    
    def predict(self,datapoint):
        return np.argmax(self.getFS(datapoint))
    
    @staticmethod
    def myrange(i,j):
        for _i in xrange(i):
            for _j in xrange(j):
                yield _i



if __name__ == '__main__':
    ################################################################################
    #setting argument
    parser = argparse.ArgumentParser(description='simple FRBS demo program for classification problem') 
    parser.add_argument('-f',required = True)
    parser.add_argument('-k',type=int,default = 2)
    args = parser.parse_args()
    infile_path = args.f
    rulesperclass = args.k
    ################################################################################
    #read input training file and show some information
    print("input file: %s"% (os.path.basename(infile_path)))
    print("rules per class: %d"% (rulesperclass))
    df=pd.read_csv(infile_path, sep=',')
    raw_data = df.drop("CLASS", axis=1).values
    data_target_names = df.CLASS.unique()
    print("number of class: %d"% (len(data_target_names)))
    print("number of fuzzy rules: %d"% (len(data_target_names)*rulesperclass))
    print("number of instance: %d"% (len(df.index)))
    print("-"*80)
    ################################################################################
    #maping category class label to numerical
    target_mapping = dict()
    for i,j in enumerate(data_target_names):
        target_mapping[j] = i
    data_target = np.asarray(map(lambda x: target_mapping.get(x) if target_mapping.has_key(x) else -1,df.CLASS))	
    ################################################################################
    #Scaling inupt data set
    scaler = preprocessing.StandardScaler().fit(raw_data)
    data_scaled = scaler.transform(raw_data)
    ################################################################################
    #build fuzzy system and predict
    R1 = FuzzyRuleBase(data_scaled,data_target,len(data_target_names),rulesperclass)
    myans = map(R1.predict,data_scaled)
    ################################################################################
    #output confusion matrix and accuracy
    cm = confusion_matrix(data_target, myans)
    print("Confusion Matrix")
    print(cm)
    print("-"*80)
    print("Accuracy:%2.2f" % (np.trace(cm)/(cm.sum()+0.0)))