
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath('.'), '../../')))
from keras.models import load_model
from src.limit import limitUsage
limitUsage("3")


# In[12]:


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




# In[13]:


train_x, train_y, val_x, val_y = get_train_data('../../data/processed_faceexp-comparison-data-train-public.csv')


# In[14]:


from keras_contrib.applications import DenseNet
import keras.backend as K
import keras 
from keras.models import Model
from keras.layers import AveragePooling2D, Dense, Flatten,BatchNormalization, Input, Lambda, Concatenate, concatenate
import keras.optimizers as optimizer
from src.config.staticConfig import StaticConfig

# Logic of the loss function
#     if label == 1:
#         return K.maximum(0.0, K.square(y-z) - K.square(y-x) + delta) + K.maximum(0.0, K.square(y-z) - K.square(z- x) + delta) 
#     elif label == 2:
#         return K.maximum(0.0, K.square(x-z) - K.square( x- y) + delta) + K.maximum(0.0, K.square(x - z) - K.square(y- z) + delta)
    
#     else:
#         return K.maximum(0.0, K.square(x- y) - K.square( x -z) + delta) + K.maximum(0.0, K.square(x- y) - K.square( y- z ) + delta)
    
def data_generator_threaded(train_x, train_y, index, folder, batch_size):
    file_name = os.path.join(folder, 'batch_'+str(index))
    if(os.path.exists(file_name)):
        return 
    row_count = train_x.shape[0]
#     indexes = np.random.randint( 0, row_count, size = batch_size )
    indexes = [x for x in range(index*batch_size, min(train_x.shape[0], (index+1)*batch_size))]
    #print(indexes)
    batch_x = [[] for x in range(3)]
    batch_y = train_y.iloc[indexes].values
    batch_y = np.reshape(batch_y, (-1,1))
    for row in train_x.iloc[indexes].values:
        #imgs = []
        i = 0
        for file in row.tolist()[16:19]:
            batch_x[i].append(cv2.imread(file))
            i+=1
            #batch_x.append(imgs)
        #print(row[16])
    file_name = os.path.join(folder, 'batch_'+str(index))
    np.savez( file_name, x =[np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)], y = batch_y)
    return 1
    


import time
def process_data_creator(train_x, train_y,temp_folder, start, max_iteration, steps = 100, batch_size =128 ):
    try:
        os.mkdir(temp_folder)
    except:
        ## incase of first time
        pass
#     os.mkdir(temp_folder)
    #cur_iteration = 0
    #max_iteration = int(train_x.shape[0]/batch_size)
    for cur_iteration in range(start, max_iteration):
        ## create 200 files
        with  ThreadPoolExecutor(max_workers= 10) as pool:
            for i in range(200):
                if(max_iteration == cur_iteration):
                    break
                future = pool.submit(data_generator_threaded, train_x, train_y, cur_iteration, temp_folder, batch_size)
                cur_iteration += 1
            pool.shutdown(wait = True)  
        #time.sleep(60)
    
from multiprocessing import Process        
def data_generator_2(train_x, train_y, steps = 100, batch_size = 128):
    
    cur_batch = 0
    # with  ProcessPoolExecutor(max_workers= 10) as pool:
    folders = ['./temp/val_batch_backup', './temp/val_batch_main']
    f_i = 0
    main_folder = folders[1]
    backup_folder = folders[0]
    max_iteration = int(train_x.shape[0]/batch_size)
    ps = []
    cur_iteration = 0
    while max_iteration < cur_iteration: 
        process = Process(target = process_data_creator, args= (train_x, train_y, main_folder,cur_iteration, 
        min(max_iteration, cur_iteration + 500)))
        cur_iteration += 500
        ps.append(process)
    
    for process in ps:
        process.join()
    #print("hellooooooooooo1")
    process.join()

# In[15]:




