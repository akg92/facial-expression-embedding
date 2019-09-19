
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath('.'), '../../')))
from keras.models import load_model
from src.limit import limitUsage
limitUsage("3")


# In[2]:


import pandas as pd
import random
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import threading
import shutil

def get_train_data(train_file):
    ## 20% for testing
    df = pd.read_csv(train_file)
    n =int(df.shape[0]*.8)
    label_train = df.iloc[:n,-1]
    df_train = df.iloc[:n,:]
    df_test = df.iloc[n:,:]
    label_test = df.iloc[n:,-1]
    return df_train, label_train, df_test, label_test


# In[3]:


train_x, train_y, val_x, val_y = get_train_data('../../data/processed_faceexp-comparison-data-train-public.csv')


# In[4]:


from keras_contrib.applications import DenseNet
import keras.backend as K
import keras 
from keras.models import Model
from keras.layers import AveragePooling2D, Dense, Flatten,BatchNormalization, Input, Lambda, Concatenate, concatenate
import keras.optimizers as optimizer

# Logic of the loss function
#     if label == 1:
#         return K.maximum(0.0, K.square(y-z) - K.square(y-x) + delta) + K.maximum(0.0, K.square(y-z) - K.square(z- x) + delta) 
#     elif label == 2:
#         return K.maximum(0.0, K.square(x-z) - K.square( x- y) + delta) + K.maximum(0.0, K.square(x - z) - K.square(y- z) + delta)
    
#     else:
#         return K.maximum(0.0, K.square(x- y) - K.square( x -z) + delta) + K.maximum(0.0, K.square(x- y) - K.square( y- z ) + delta)
    
def loss_fun(label, pred, delta = 0.001):
    ## label is of the form [first_pred, second_pred, last_pred]
    #i label means the i is different from the rest of the pair
    x,y,z  = pred[:,0], pred[:,1], pred[:,2]
    n = pred.shape[0]
    ## y and z is similar
    l_1 = K.maximum(0.0, K.square(y-z) - K.square(y-x) + delta) + K.maximum(0.0, K.square(y-z) - K.square(z- x) + delta)
    l_2 = K.maximum(0.0, K.square(x-z) - K.square( x- y) + delta) + K.maximum(0.0, K.square(x - z) - K.square(y- z) + delta)
    l_3  = K.maximum(0.0, K.square(x- y) - K.square( x -z) + delta) + K.maximum(0.0, K.square(x- y) - K.square( y- z ) + delta)
    
    l_1_eq = K.cast(K.equal(label, 1),'float32')
    l_2_eq = K.cast(K.equal(label, 2), 'float32')
    l_3_eq = K.cast(K.equal(label, 3), 'float32')
    
    return K.mean(K.sum([l_1*l_1_eq , l_2 * l_2_eq*l_2 , l_3_eq * l_3], axis = 1 ))

        
def accuracy_c(label, pred, delta = 0.0):
    x,y,z  = pred[:,0], pred[:,1], pred[:,2]
    n = pred.shape[0]
    ## y and z is similar
    l_1 = K.maximum(0.0, K.square(y-z) - K.square(y-x) + delta) + K.maximum(0.0, K.square(y-z) - K.square(z- x) + delta)
    l_2 = K.maximum(0.0, K.square(x-z) - K.square( x- y) + delta) + K.maximum(0.0, K.square(x - z) - K.square(y- z) + delta)
    l_3  = K.maximum(0.0, K.square(x- y) - K.square( x -z) + delta) + K.maximum(0.0, K.square(x- y) - K.square( y- z ) + delta)
    
    l_1_eq = K.cast(K.equal(label, 1),'float32')
    l_2_eq = K.cast(K.equal(label, 2), 'float32')
    l_3_eq = K.cast(K.equal(label, 3), 'float32')
    s = K.sum([l_1*l_1_eq , l_2 * l_2_eq*l_2 , l_3_eq * l_3], axis = 1 )
    
    #s = loss_fun(K.cast(y_true, 'int32'), y_pred)
    return K.equal(s, 0.0)
        
        
    


# In[25]:



from keras.models import load_model
from src.config.staticConfig import StaticConfig
def split_indexes(df, batch = 512):
    n = df.shape[0]
    inds = []
    start = 0
    while(start < n):
        inds.append( (start, min(start+batch, n)))
        start += batch
    return inds
    
    
def run_validation(val_x, val_y, model_file):
    feb_model = load_model(model_file,  custom_objects=dict(loss_fun=loss_fun,accuracy_c=accuracy_c))
    
    
    ## load data
    inds = split_indexes(val_x)
    result = []
    print("Started execution")
    loss = 0
    accuracy = 0
    level = 0
    for start, end in inds:
        val_xx = val_x[start:end]
        val_yy = val_y[start:end]
        batch_x = [[] for x in range(3)]
        for row in val_xx.values:
            #imgs = []
            i = 0
            #print(row.shape)
            for file in row.tolist()[16:19]:
                file = StaticConfig.getPretrainedImageFromFile(file)
                #print(file)
                batch_x[i].append(cv2.imread(file))
                i+=1
            #print(np.stack(batch_x[0]).shape)
        result.append(feb_model.evaluate(x = [ np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0) ], y = val_yy.values))
        level += 1
        loss += result[-1][0]
        accuracy += result[-1][1]
        #print(' Loss = {}, Accuracy = {}'.format(loss/level,accuracy/level))
        #print(result[-1])
    ## validate
    print("End of Execution")
    
    ### average
    loss = 0
    accuracy = 0
    for r in result:
        loss += r[0]
        accuracy += r[1]
    return loss/len(result), accuracy/len(result) 
    
    
        


# In[19]:


#loss, accuracy = run_validation(val_x, val_y, 'weights-improvement-07-0.97.hdf5')
#print('Loss = {}, Accuracy = {}'.format(loss, accuracy))


# In[ ]:


val_x.shape


# In[ ]:



import os
import glob
def calc_all():
    with open('val_result.csv','w') as f:
        f.write('level, loss, accuracy, file_name')
        for file in glob.glob('*.hdf5'):
            try:
                loss, accuracy = run_validation(val_x, val_y, file)
                iteration = int(file.split("-")[1])
                f.write('{},{},{},{}'.format(iteration, loss, accuracy,file))
                
            except:
                print('failed '+file)
                pass
            
calc_all()

