
from mira import detectors
from mira.core import Image
import cv2
import os
from src.config.staticConfig import StaticConfig 
from src.limit import limitUsage

limitUsage("2")
det = detectors.MTCNN()
def get_out_file_name(out_dir, file_name):
    f_name = os.path.basename(file_name)
    return os.path.join(out_dir, 'processed_'+ f_name)

def cut_image(file_name, out_dir):
    out_file_name = get_out_file_name(out_dir, file_name)    
    m_image = Image.read(file_name)
    faces = det.detect(m_image)
    if( not faces or not faces[0]):
        print('face_not_found for {}'.format(file_name))
        resizedImage = cv2.resize(m_image, (160, 160))
        cv2.imwrite(out_file_name, resizedImage)
    else :
        extractedImg = faces[0].selection.extract(m_image)
        resizedImage = cv2.resize(extractedImg, (160, 160))
        print(out_file_name)
        cv2.imwrite(out_file_name, resizedImage)  


import keras.backend as K 
    
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


"""
    Pre process all the files in the folder
"""
def cut_images(folder_name, out_dir):
    
    if os.path.exists(out_dir):
        os.rmdir(out_dir)
    os.mkdir(out_dir)
    print('Foldername = '+ folder_name)
    for file in os.listdir(folder_name):
        print('file {} = '+ file)
        cut_image(os.path.join(folder_name, file) , out_dir)
import numpy as np

def predict(model_obj, img_file):
    img = cv2.imread(img_file)
    rep = model_obj.predict([ np.array([img]), np.array([img]), np.array([img]) ] )
    return rep[0]

"""
compute similarity
"""
def compute_most_similar(file_ids, result):

    n  = len(file_ids)
    similarity = []
    for i in range( n):
        distance = np.zeros(n)
        for j in range(n):
            distance[j] = np.linalg.norm(result[i] - result[j])
        
        pos = np.argsort(distance)
        s = [ file_ids[i] ] 
        for p in pos:
            if( p != i):
                s.append(file_ids[p])
        similarity.append(s)
    return similarity
                   



from keras.models import load_model
import pandas as pd
def rep_all(model_file , in_dir, out_dir, result_folder_path ):
    
    cut_images(in_dir , out_dir)
    print(model_file)
    loaded_model  = load_model(model_file, custom_objects=dict(loss_fun=loss_fun,accuracy_c=accuracy_c) )

    result = []
    file_ids = []
    for file_name in os.listdir(in_dir):
        processed_file = get_out_file_name(out_dir, file_name)
        result.append ( predict(loaded_model, processed_file))
        file_ids.append(file_name)
    
    similarity = compute_most_similar(file_ids, result)
    result = np.column_stack([file_ids,result])
    pd.DataFrame(result).to_csv(os.path.join(result_folder_path, 'prediction.csv'))
    pd.DataFrame(similarity).to_csv(os.path.join( result_folder_path, 'similar.csv'))
    
    #np.savetxt( os.path.join(result_folder_path, 'prediction.csv'), result, delimiter=',')
    #np.savetxt( os.path.join( result_folder_path, 'similar.csv'), np.array(similarity), delimiter = ',' )

import sys
model_file = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3] 
result_file_path = sys.argv[4]

rep_all(model_file, in_dir, out_dir, result_file_path)



    
