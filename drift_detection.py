import numpy as np
import matplotlib.pyplot as plt
from river import drift
from river import metrics
from river import ensemble
from river import dummy
from river import tree
from river import stream

filePath = './data/cpu.arff'

class ModelFactory:
    def createModel(modelType):
        #Hoeffdingtrees
        if modelType=='HT':
            return tree.HoeffdingTreeClassifier()
        #Adaptiverandomforest
        if modelType=='ARF':
            return ensemble.AdaptiveRandomForestClassifier()
        #StreamingRandomPatches
        if modelType=='SRP':
            return ensemble.SRPClassifier()
        #NoChangeClassifier
        if modelType=='NoChange':
            return dummy.NoChangeClassifier()
        #MajorityClassclassifier
        if modelType=='MajorityClass':
            return dummy.PriorClassifier()

modelsToDevelop=['HT']
showInlinePlots=True
window=50
adwin=drift.ADWIN()
kswin=drift.KSWIN()
HDDMA=drift.HDDM_A()
HDDMW=drift.HDDM_W()
plt.figure(figsize=(6,3))
plt.xlabel('Examples')
plt.ylabel('Accuracy[%]')

plotName=''
for modelType in modelsToDevelop:
    accuracySeries=[]
    plotName=modelType+''+plotName
    instanceCount=0
    myStream=stream.iter_arff(filePath,target='class')

    model=ModelFactory.createModel(modelType)
    metric=metrics.ClassificationReport()
    for x,y in myStream:
        ypred=model.predict_one(x)
        wrongPrediction = 0 if ypred==y else 1

        adwin.update(wrongPrediction)
        kswin.update(wrongPrediction)
        HDDMA.update(wrongPrediction)
        HDDMW.update(wrongPrediction)

        if adwin.drift_detected:
            plt.axvline(x=instanceCount,color='r')
        if kswin.drift_detected:
            plt.axvline(x=instanceCount,color='b')
        if HDDMA.drift_detected:
            plt.axvline(x=instanceCount,color='k')
        if HDDMW.drift_detected:
            plt.axvline(x=instanceCount,color='g')

        model.learn_one(x,y)
        if ypred is not None:
            metric.update(y,ypred)
            accuracySeries.append(100*(1-wrongPrediction))
        else:
            accuracySeries.append(np.nan)

        instanceCount=instanceCount+1
    instanceIndexes=np.arange(1,instanceCount+1)

    slidingWindowAccuracy=[]
    for ind in range(len(accuracySeries)-window+1):
        slidingWindowAccuracy.append(np.mean(accuracySeries[ind:ind+window]))
    for ind in range(window-1):
        slidingWindowAccuracy.insert(0,np.nan)
    # plt.plot(instanceIndexes[::5],slidingWindowAccuracy[::5],label=modelType)
    plt.plot(slidingWindowAccuracy,label=modelType)
plt.legend()
plt.show()
    