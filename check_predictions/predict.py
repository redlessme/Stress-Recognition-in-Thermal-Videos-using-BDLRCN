import pandas as pd
import numpy as np
import pickle
# This script is used to evaluation on test set

pd.set_option('display.max_rows', 1000)
df = pd.read_pickle('bilstm_prediction.pkl')
###uncomment line 8 and cmment line 6 to test our model withough bidirectional training.
# df = pd.read_pickle('lstm_prediction.pkl')
y=df['y'].tolist()
y_pred=df['y_pred'].tolist()
y_predd=[]
y_gt=[]
for i in range(len(y_pred)):
    if  y_pred[i]==1:
        y_predd.append('Stressful')
    else:
        y_predd.append('Calm')
for j in y:
    if 'Stressful' in j:
        y_gt.append('Stressful')
    else:
        y_gt.append('Calm')


N=len(y_gt)

mistake=0
for k in range(N):
    if y_gt[k]!=y_predd[k]:
        mistake+=1
acc=(N-mistake)/N
print('Total test videos are ',N)
print('y',y_gt)
print('y_pred',y_predd)
print('Accuracy in the test set is ',acc)