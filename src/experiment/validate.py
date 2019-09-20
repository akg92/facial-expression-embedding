
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
from src.config.staticConfig import StaticConfig

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
def process_data_creator(train_x, train_y,temp_folder, steps = 100, batch_size =128 ):
    try:
        os.mkdir(temp_folder)
    except:
        ## incase of first time
        pass
#     os.mkdir(temp_folder)
    #cur_iteration = 0
    max_iteration = int(train_x.shape[0]/batch_size)
    for cur_iteration in range(max_iteration):
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
    process = Process(target = process_data_creator, args= (train_x, train_y, main_folder, steps, batch_size))
    process.start()
    time.sleep(60)
    #print("hellooooooooooo1")
    
    while True:        
        batch_file_name = os.path.join(main_folder, 'batch_'+str(cur_batch)+'.npz')
        while(not os.path.exists(batch_file_name)):
            pass
        npzfile = np.load(batch_file_name, allow_pickle= True)
        cur_batch = cur_batch + 1  
        #print(batch_file_name)
        ## remove used file
        #os.remove(batch_file_name)
        #print("helloooooooooooo")
        #print('{}:{}:{}'.format(npzfile['x'][0].shape,npzfile['x'][1].shape, npzfile['x'][2].shape))
        yield ([npzfile['x'][0], npzfile['x'][1], npzfile['x'][2]],npzfile['y'] )

    process.join()
    
    
def data_generator(train_x, train_y, steps = 100, batch_size = 128):
    
    cur_batch = 0
    # with  ProcessPoolExecutor(max_workers= 10) as pool:
    folders = ['./temp/val_batch_backup', './temp/val_batch_main']
    f_i = 0
    main_folder = folders[1]
    backup_folder = folders[0]
#     process = Process(target = process_data_creator, args= (train_x, train_y, main_folder, steps, batch_size))
#     process.start()
    #time.sleep(60)
    #print("hellooooooooooo1")
    md = int(train_x.shape[0]/batch_size)
    x, y = None, None
    while True:        
        batch_file_name = os.path.join(main_folder, 'batch_'+str(cur_batch)+'.npz')
        print(batch_file_name)
        while(not os.path.exists(batch_file_name)):
            print("skipped" + batch_file_name)
            cur_batch = (cur_batch + 1)%md
            continue
        npzfile = None
        try:
            cur_batch = (cur_batch + 1)%md
            with np.load(batch_file_name, allow_pickle = True) as npzfile:
                x = np.copy(npzfile['x'])
                y = np.copy(npzfile['y'])
        except:
            continue
            
        #cur_batch = (cur_batch + 1)%md
        
        yield ([x[0],x[1],x[2]],y )
            
#         try:
#             npzfile =  np.load(batch_file_name)
#             if(npzfile['x'][0].shape[0] != batch_size or npzfile['y'].shape[0] != batch_size):
#                 cur_batch = (cur_batch + 1)%md
#                 print("skipped" + batch_file_name)
                
#         except:
#             cur_batch = (cur_batch + 1)%md
#             continue
#         cur_batch = (cur_batch + 1)%md
        
        
            
        #print(batch_file_name)
        ## remove used file
        #os.remove(batch_file_name)
        #print("helloooooooooooo")
        #print('{}:{}:{}'.format(npzfile['x'][0].shape,npzfile['x'][1].shape, npzfile['x'][2].shape))
        #yield ([npzfile['x'][0], npzfile['x'][1], npzfile['x'][2]],npzfile['y'] )
     


    

    


# In[5]:



from keras.models import load_model
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
#     inds = split_indexes(val_x)
#     result = []
#     print("Started execution")
#     loss = 0
#     accuracy = 0
#     level = 0
#     for start, end in inds:
#         val_xx = val_x[start:end]
#         val_yy = val_y[start:end]
#         batch_x = [[] for x in range(3)]
#         for row in val_xx.values:
#             #imgs = []
#             i = 0
#             #print(row.shape)
#             for file in row.tolist()[16:19]:
#                 file = StaticConfig.getPretrainedImageFromFile(file)
#                 #print(file)
#                 batch_x[i].append(cv2.imread(file))
#                 i+=1
#             #print(np.stack(batch_x[0]).shape)

    
    return feb_model.evaluate_generator(data_generator(val_x, val_y), steps = int(val_x.shape[0]/128))
#     level += 1
#         loss += result[-1][0]
#         accuracy += result[-1][1]
#         #print(' Loss = {}, Accuracy = {}'.format(loss/level,accuracy/level))
#         #print(result[-1])
#     ## validate
#     print("End of Execution")
    
#     ### average
#     loss = 0
#     accuracy = 0
#     for r in result:
#         loss += r[0]
#         accuracy += r[1]
#     return loss/len(result), accuracy/len(result) 
    
    
        


# In[ ]:


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


# In[ ]:


val_x.shape

