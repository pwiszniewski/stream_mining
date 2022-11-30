from river import ensemble
from river import dummy
from river import tree
from river import stream
from river import metrics

# filePath = './data/weather.arff'
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


modelsToDevelop=['NoChange','MajorityClass', 'HT']
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(6,3))
plt.xlabel('Examples')
plt.ylabel('Accuracy[%]')
for modelType in modelsToDevelop:
    accuracySeries=[]
    instanceCount=0
    myStream=stream.iter_arff(filePath,target='class')
    model=ModelFactory.createModel(modelType)
    metric=metrics.ClassificationReport()
    for x,y in myStream:
        ypred=model.predict_one(x)
        model.learn_one(x,y)
        if ypred is not None:
            metric.update(y,ypred)
            accuracySeries.append(100*metric._accuracy.get())
        else:
            accuracySeries.append(np.nan)
            instanceCount=instanceCount+1
    instanceIndexes=np.arange(1,instanceCount+1)
    # plt.plot(instanceIndexes[::5],accuracySeries[::5],label=modelType)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(accuracySeries,label=modelType)

    window = 30
    slidingWindowAccuracy=[]
    for ind in range(len(accuracySeries)-window+1):
        slidingWindowAccuracy.append(np.mean(accuracySeries[ind:ind+window ] ) )
        #initiatefirstwindow-1valueswithnansaswecalculating
        #movingaverageoutofwindowvaluesyet
        for ind in range(window-1):
            slidingWindowAccuracy.insert(0,np.nan)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(slidingWindowAccuracy, label='sliding window')
plt.legend()
plt.show()