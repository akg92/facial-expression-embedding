
# coding: utf-8

# In[16]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath('.'), '../../')))
from keras.models import load_model
from src.limit import limitUsage
limitUsage("2")


# In[17]:


modelFile = '../model/keras/model/facenet_keras.h5'
# faceNetModel = load_model(modelFile)


# In[18]:


#faceNetModel.summary()


# In[19]:


facenet_model = load_model(modelFile)


# In[20]:


#!pip install git+https://www.github.com/keras-team/keras-contrib.git
#!pwd


# In[42]:


import pandas as pd
import random
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import threading
from src.config.staticConfig import StaticConfig
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
def temp_generator(arr):
    for ele in arr:
        yield ele
        


def split_class_variation(x, y ):
    labels = [ 'ONE_CLASS_TRIPLET', 
        'TWO_CLASS_TRIPLET','THREE_CLASS_TRIPLET']
    xs = []
    ys = []
    for label in labels:
        x_x =  x[ x[15] ==label].index.tolist()
        #y_y =  y[ x[15] == label].index.tolist()
        xs.append(x_x)
        ys.append( x_x)

    return xs, ys

def generate_indexes(xs, batch_size):
    splits =[ batch_size//3, batch_size//3, batch_size - (batch_size//3 ) *2]

    indexes = []
    for i in range(3):
        indexes.extend(np.random.choice(xs[i],splits[i]))
    
    return indexes
    
def data_generator_balanced(train_x, train_y, batch_size):
    row_count = train_x.shape[0]
    xs, ys = split_class_variation(train_x, train_y)
    while True:
        indexes = generate_indexes(xs, batch_size)
        #print(indexes)
        batch_x = [[] for x in range(3)]
        batch_y = train_y.iloc[indexes].values
        batch_y = np.reshape(batch_y, (-1,1))
        try:
            for row in train_x.iloc[indexes].values:
                i = 0
                for file in row.tolist()[16:19]:
                    file = StaticConfig.getPretrainedImageFromFile(file)
                    #print(file)
                    batch_x[i].append(cv2.imread(file))
                    i+=1
        except:
            continue
        yield [np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)],batch_y






def data_generator_threaded(train_x, train_y, batch_size):
    row_count = train_x.shape[0]
    while True:
        indexes = np.random.randint( 0, row_count, size = batch_size )
        #print(indexes)
        batch_x = [[] for x in range(3)]
        batch_y = train_y.iloc[indexes].values
        batch_y = np.reshape(batch_y, (-1,1))
        try:
            for row in train_x.iloc[indexes].values:
                i = 0
                for file in row.tolist()[16:19]:
                    file = StaticConfig.getPretrainedImageFromFile(file)
                    #print(file)
                    batch_x[i].append(cv2.imread(file))
                    i+=1
        except:
            continue
        yield [np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)],batch_y
    



def validation_generator(x, y, batch_size = 48):
    size = x.shape[0]
    cur_index = 0
    while True:
        indexes = []
        for i in range(batch_size):
            indexes.append(cur_index)
            cur_index = (cur_index+1)%size
            
        batch_x = [[] for x in range(3)]
        batch_y = train_y.iloc[indexes].values
        batch_y = np.reshape(batch_y, (-1,1))
        ## fetch images. image indexes are 
        for row in train_x.iloc[indexes].values:
            #imgs = []
            i = 0
            for file in row.tolist()[16:19]:
                file = StaticConfig.getPretrainedImageFromFile(file)
                #print(file)
                batch_x[i].append(cv2.imread(file))
                i+=1
            #batch_x.append(imgs)
        #print('{}:{}:{}'.format(npzfile['x'][0].shape,npzfile['x'][1].shape, npzfile['x'][2].shape))
        yield [np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)], batch_y
    
            

from keras_contrib.applications import DenseNet
import keras.backend as K
import keras 
from keras.models import Model
from keras.layers import AveragePooling2D, Dense, Flatten,BatchNormalization, Input, Lambda, Concatenate, concatenate, Dropout
import keras.optimizers as optimizer

def loss_fun(label, pred, delta = 0.001):
    x,y,z  = pred[:,0], pred[:,1], pred[:,2]
    n = pred.shape[0]

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

    l_1 = K.maximum(0.0, K.square(y-z) - K.square(y-x) + delta) + K.maximum(0.0, K.square(y-z) - K.square(z- x) + delta)
    l_2 = K.maximum(0.0, K.square(x-z) - K.square( x- y) + delta) + K.maximum(0.0, K.square(x - z) - K.square(y- z) + delta)
    l_3  = K.maximum(0.0, K.square(x- y) - K.square( x -z) + delta) + K.maximum(0.0, K.square(x- y) - K.square( y- z ) + delta)
    
    l_1_eq = K.cast(K.equal(label, 1),'float32')
    l_2_eq = K.cast(K.equal(label, 2), 'float32')
    l_3_eq = K.cast(K.equal(label, 3), 'float32')
    s = K.sum([l_1*l_1_eq , l_2 * l_2_eq*l_2 , l_3_eq * l_3], axis = 1 )
    
    return K.equal(s, 0.0)
        
        
    

def create_new_model(input_shape):
    inputs = Input(shape=input_shape)

    model1 = Model(input=facenet_model.get_input_at(0) , output = facenet_model.get_layer('add_15').output)
    for layer in model1.layers:
        layer.trainable = False
    
    inter_shape =  model1.output_shape[1:]

    dense = DenseNet(
                     input_shape = inter_shape, dropout_rate=0.2, nb_filter = 600, include_top = False,  depth = 1, nb_dense_block = 2, growth_rate = 32)(model1.output)

    x = AveragePooling2D(pool_size=(4,4), name='cutome_avg_pool_1')(dense)
    x = Flatten(name  = 'cutome_flatten_2')(x)
    x = Dense(512, activation='relu', name = 'custome_dense_dense_3')(x)
    x = BatchNormalization(name='cutome_batch_normalization_4')(x)
    x = Dropout(0.2)(x)
    x = Dense(20, activation='relu', name = 'custome_dense_5')(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'custome_l2_normalize_6')(x)
    
    model = Model(inputs=[model1.input], outputs=x)
    input_1 = Input( shape = input_shape)
    input_2 = Input( shape = input_shape)
    input_3 = Input( shape = input_shape)
    
    l1 = model(input_1)
    l2 = model(input_2)
    l3 = model(input_3)
    
    x = concatenate([l1, l2, l3])
    print(x.shape)
    final_model = Model(inputs=[input_1, input_2, input_3], outputs=x)
    
    
    final_model.compile(optimizer=optimizer.Adam( lr = 0.0005),
              loss= loss_fun, metrics = [accuracy_c] )
    return final_model
    


embedding_model = create_new_model((160,160,3))
embedding_model.summary()




train_x, train_y, val_x, val_y = get_train_data('../../data/processed_faceexp-comparison-data-train-public.csv')



from keras.callbacks import ModelCheckpoint
batch_size = 120
filepath="./dropout_balanced/weights-improvement-{epoch:02d}-{val_accuracy_c:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=False)
callbacks_list = [checkpoint]
samples_per_epoch= 100
history = embedding_model.fit_generator(data_generator_balanced(train_x, train_y, batch_size),
    samples_per_epoch= samples_per_epoch,callbacks=callbacks_list,
    validation_data = data_generator_threaded(val_x, val_y, batch_size), validation_steps= 20, 
    nb_epoch=50, workers= 5, use_multiprocessing= True, max_queue_size=25 )



import pickle
with open('trainHistoryDict_balanced', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

