from river import drift
import numpy as np
import matplotlib.pyplot as plt

dataStreamIncremental = np.concatenate([np.full(200,2),np.arange(2,3,1/600),np.full(1200,3)])
dataStreamSudden = np.concatenate([500*[0], np.full(1800,3)])
dataStreamReoccurring = np.concatenate([500*[0], np.full(500,3), 1300*[0]])
dataStreamGradual = np.concatenate([300*[0], np.full(50,3), 300*[0], np.full(150,3), 400*[0], np.full(750,3)])
# model = drift.ADWIN()
# model = drift.KSWIN()
model = drift.HDDM_A()
# datastream = dataStreamIncremental
# datastream = dataStreamSudden
# datastream = dataStreamReoccurring
datastream = dataStreamGradual
plt.figure(figsize=(12,3))
x = list(range(len(datastream)))[::20]
y = datastream[::20],
plt.scatter(x, y, s=5);
plt.xlabel('Time')
#adaptedfromriverexample
for i,val in enumerate(datastream):
    _ = model.update(val)
    if model.drift_detected:
        print(f'Change detected at example{i},input value at which change was detected:{val}')
        plt.axvline(x=i,color='r')
plt.show()