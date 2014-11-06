from sklearn import datasets, cluster, preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np

class FuzzyRuleBase:

    def __init__(self, data, target,numofclass,numofrulesperclass):
        self.nc = numofclass
        self.nrpc = numofrulesperclass
        self.nr = numofclass*numofrulesperclass
        self.dim = data.shape[-1]
        self.X = (map(lambda i: data[target == i],xrange(self.nc)))
        self.rule_target = np.arange( self.nr ).reshape((self.nc,self.nrpc))
        Y = (map(self.getcluster, xrange(numofclass)))
        M = np.asarray(map(lambda i,j:self.X[i][Y[i]==j].mean(axis=0),self.myrange(self.nc,self.nrpc ), range(self.nrpc )*self.nc))
        S = np.asarray(map(lambda i,j:self.X[i][Y[i]==j].std(axis=0),self.myrange(self.nc,self.nrpc ), range(self.nrpc )*self.nc))
        self.m = M
        self.s = S
        self.s2 = np.square(S)

    def getcluster(self,k):
        k_means = cluster.KMeans(self.nrpc)
        k_means.fit(self.X[k])
        return k_means.labels_

    def myrange(self,i,j):
        for _i in xrange(i):
            for _j in xrange(j):
                yield _i

    def gaussianMF(self,datapoint):
        #mfv = np.ones((self.nc, self.dim))
        mfv = np.subtract(datapoint,self.m)
        np.square(mfv,mfv)
        np.negative(mfv,mfv)
        np.divide(mfv,self.s2,mfv)
        np.exp(mfv,mfv)
        #mfv = np.exp(np.divide(np.negative(np.square(np.subtract(datapoint,model.m))),np.square(model.s))
        return mfv               
    
    def getsoftFS(self,datapoint):
        fss = np.prod(self.gaussianMF(datapoint), axis=1)
        fss = np.asarray(map(lambda x:FuzzyRuleBase.softMAX(fss[self.rule_target[x]],50),xrange(self.nc)))
        return fss

    def getFS(self,datapoint):
        fss = np.prod(self.gaussianMF(datapoint), axis=1)
        fss = np.asarray(map(lambda x:(fss[self.rule_target[x]]).max(),xrange(self.nc)))
        return fss
    def predict(self,datapoint):
    	return np.argmax(self.getFS(datapoint))

    def productTNOME(mfv):
        if mfv.ndim == 1:
            fs = reduce((lambda x,y: x*y), mfv)
        return fs
    @staticmethod
    def softMAX(alpha,q):
        denominator = np.exp(alpha*q)
        numerator = alpha*denominator
        return numerator.sum()/denominator.sum()

if __name__ == '__main__':
    iris = datasets.load_iris()
    scaler = preprocessing.StandardScaler().fit(iris.data)
    iris_scaled = scaler.transform(iris.data)

    R1 = FuzzyRuleBase(iris_scaled,iris.target,len(iris.target_names),3)
    
    #print((R1.s))
    print("-"*80)
    
    myans = map(R1.predict,iris_scaled)
        
    cm = confusion_matrix(iris.target, myans)
    print(cm)
