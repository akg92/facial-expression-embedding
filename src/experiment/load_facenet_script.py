
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath('.'), '../../')))
from keras.models import load_model
from src.limit import limitUsage
limitUsage("5")


# In[ ]:


modelFile = '../model/keras/model/facenet_keras.h5'
# faceNetModel = load_model(modelFile)


# In[ ]:


#faceNetModel.summary()


# In[ ]:


facenet_model = load_model(modelFile)


# In[ ]:


#!pip install git+https://www.github.com/keras-team/keras-contrib.git
#get_ipython().system('pwd')


# In[ ]:


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
        
def data_generator_threaded(train_x, train_y, index, folder, batch_size):
    row_count = train_x.shape[0]
    indexes = np.random.randint( 0, row_count, size = batch_size )
    #print(indexes)
    batch_x = [[] for x in range(3)]
    batch_y = train_y.iloc[indexes].values
    batch_y = np.reshape(batch_y, (-1,1))
    for row in train_x.iloc[indexes].values:
        #imgs = []
        i = 0
        for file in row.tolist()[16:19]:
            file = StaticConfig.getPretrainedImageFromFile(file)
            #print(file)
            batch_x[i].append(cv2.imread(file))
            i+=1
            #batch_x.append(imgs)
        #print(row[16])
    file_name = os.path.join(folder, 'batch_'+str(index))
    np.savez( file_name, x =[np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)], y = batch_y)
    return 1
    


def data_generator(train_x, train_y, steps = 100, batch_size = 48):
    
    cur_batch = 0
    while True:
        if(cur_batch == 0):
            if(os.path.exists('train_batch_backup')):
                shutil.rmtree('train_batch_main')
                os.rename('train_batch_backup', 'train_batch_main')
                os.mkdir('train_batch_backup')
            else:
                
                try:
                    os.mkdir('train_batch_backup')
                    os.mkdir('train_batch_main')
                except Exception as  e:
                    #print(e)
                    #print('error in create folder')
                    pass
                    
                #data_generator_threaded(train_x, train_y, cur_batch, 'train_batch_main', batch_size)
                with  ThreadPoolExecutor(max_workers= 10) as pool_t:
                    for i in range(steps):
                        future = pool_t.submit(data_generator_threaded, train_x, train_y, i, 'train_batch_main', batch_size)
                        #print(future.result())
                        #data_generator_threaded(train_x, train_y, i, 'train_batch_main', batch_size)
                    pool_t.shutdown(True)
                    #print(pool_t.result())
            
            
            with  ThreadPoolExecutor(max_workers= 10) as pool:
                for i in range(steps):
                    pool.submit(data_generator_threaded, train_x, train_y, i, 'train_batch_backup', batch_size)
                    #data_generator_threaded(train_x, train_y, i, 'train_batch_backup', batch_size)
                pool.shutdown(True)
                
        batch_file_name = os.path.join('train_batch_main', 'batch_'+str(cur_batch)+'.npz')
        while(not os.path.exists(batch_file_name)):
            pass
        npzfile = np.load(batch_file_name)
        cur_batch = (cur_batch+1) % steps
        yield [npzfile['x'][0], npzfile['x'][1], npzfile['x'][2]],npzfile['y'] 
            
                
    
# def data_generator(train_x, train_y, batch_size = 48):
#     row_count = train_x.shape[0]
#     while True:
#         indexes = np.random.randint( 0, row_count, size = batch_size )
#         #print(indexes)
#         batch_x = [[] for x in range(3)]
#         batch_y = train_y.iloc[indexes].values
#         batch_y = np.reshape(batch_y, (-1,1))
#         ## fetch images. image indexes are 
#         for row in train_x.iloc[indexes].values:
#             #imgs = []
#             i = 0
#             for file in row.tolist()[16:19]:
#                 file = StaticConfig.getPretrainedImageFromFile(file)
#                 #print(file)
#                 batch_x[i].append(cv2.imread(file))
#                 i+=1
#             #batch_x.append(imgs)
#         yield [np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)], batch_y
            
def validation_generator(x, y, batch_size = 48):
    size = x.shape[0]
    cur_index = 0
    while True:
        indexes = []
        for i in range(batch_size):
            indexes.append(cur_index)
            cur_index = (cur_index+1)%size
            
        batch_x = [[] for x in range(3)]
        batch_y = train_y.iloc[indexes,-1].values
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
        yield [np.stack(batch_x[0], axis =0), np.stack(batch_x[1], axis =0), np.stack(batch_x[2], axis =0)], batch_y
    
            
            
        
    


# In[ ]:


#print(train_x.iloc[[1,2]])
#t1,t2 =next(data_generator(train_x, train_y, 1000, 64))
#print(t2.shape)
#print(t2)


# In[ ]:


from keras_contrib.applications import DenseNet
import keras.backend as K
import keras 
from keras.models import Model
from keras.layers import AveragePooling2D, Dense, Flatten,BatchNormalization, Input, Lambda, Concatenate
import keras.optimizers as optimizer

def loss_fun(label, pred, delta = 0.001):
    ## label is of the form [first_pred, second_pred, last_pred]
    #i label means the i is different from the rest of the pair
    x,y,z  = pred[0], pred[1], pred[2]
    
    ## y and z is similar
    if label == 1:
        return K.maximum(0.0, K.square(y-z) - K.square(y-x) + delta) + K.maximum(0.0, K.square(y-z) - K.square(z- x) + delta) 
    elif label == 2:
        return K.maximum(0.0, K.square(x-z) - K.square( x- y) + delta) + K.maximum(0.0, K.square(x - z) - K.square(y- z) + delta)
    
    else:
        return K.maximum(0.0, K.square(x- y) - K.square( x -z) + delta) + K.maximum(0.0, K.square(x- y) - K.square( y- z ) + delta)
    
        
def accuracy_c(y_true, y_pred):
    s = loss_fun(K.cast(y_true, 'int32'), y_pred)
    return K.equal(s, 0.0)
        
        
    

def create_new_model(input_shape):
#     facenet_model = load_model(modelFile)
    inputs = Input(shape=input_shape)
    #i = facenet_model(inputs)
    #print(facenet_model.get_input_at(0).shape)
    model1 = Model(input=facenet_model.get_input_at(0) , output = facenet_model.get_layer('add_15').output)
    for layer in model1.layers:
        layer.trainable = False
    
    inter_shape =  model1.output_shape[1:]
    print(inter_shape)
    #model2 = Model(input=model1.input1, outputs = inter_layer);
    #inter_shape = 
    print(inter_shape)
    dense = DenseNet(
                     input_shape = inter_shape, nb_filter = 600, include_top = False,  depth = 1, nb_dense_block = 2, growth_rate = 32)(model1.output)
    #d.layers.pop()
    #d.layers.pop()
    #d.layers.pop()
    #d = d.layers.pop()
    #dense.summary()
    #model = d
    #model.summary()
    x = AveragePooling2D(pool_size=(4,4), name='cutome_avg_pool_1')(dense)
    x = Flatten(name  = 'cutome_flatten_2')(x)
    x = Dense(512, activation='relu', name = 'custome_dense_dense_3')(x)
    x = BatchNormalization(name='cutome_batch_normalization_4')(x)
    x = Dense(16, activation='relu', name = 'custome_dense_5')(x)
    #x = Dense(1, activation='relu', name = 'custome_dense_output_1')(x)
    x = Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'custome_l2_normalize_6')(x)
    #fin = d(inter_layer.output)
    
    model = Model(inputs=[model1.input], outputs=x)
    input_1 = Input( shape = input_shape)
    input_2 = Input( shape = input_shape)
    input_3 = Input( shape = input_shape)
    
    l1 = model(input_1)
    l2 = model(input_2)
    l3 = model(input_3)
    
    x = Concatenate()([l1, l2, l3])
    final_model = Model(inputs=[input_1, input_2, input_3], outputs=x)
    
    
    final_model.compile(optimizer=optimizer.Adam( lr = 0.0005),
              loss= loss_fun, metrics = [accuracy_c] )
    return final_model
    


# In[ ]:


embedding_model = create_new_model((160,160,3))


# In[ ]:


embedding_model.summary()


# In[ ]:


# example of loading an image with the Keras API
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
test_img = img_to_array(load_img('test.jpg'))
#test_imgs = (test_img, test_img, test_img)
inp = np.expand_dims(test_img, axis = 0)
embedding_model.predict([inp, inp, inp])


# In[ ]:


embedding_model.input


# In[ ]:


train_x, train_y, val_x, val_y = get_train_data('../../data/processed_faceexp-comparison-data-train-public.csv')


# In[ ]:


from keras.callbacks import ModelCheckpoint
batch_size = 64
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
samples_per_epoch= 10000
embedding_model.fit_generator(data_generator(train_x, train_y, samples_per_epoch, batch_size), validation_data = validation_generator(val_x, val_y,batch_size), validation_steps = 100 , samples_per_epoch= samples_per_epoch,callbacks=callbacks_list, nb_epoch=100)


# In[ ]:


facenet_model.summary()

