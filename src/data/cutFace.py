from mira.core import Image
import cv2
import os
from src.limit import limitUsage
limitUsage("2")

def get_out_file_name(out_dir, file_name):
    f_name = os.path.basename(file_name)
    return os.path.join(out_dir, 'processed_'+ f_name)

def cut_image(file_name, out_dir):
    out_file_name = get_out_file_name(out_dir, file_name)    
    m_image = cv2.imread(file_name)
    faces = Image.detector.detect(m_image)
    if( not faces or not faces[0]):
        print('face_not_found for {}'.format(file_name))
        resizedImage = cv2.resize(m_image, (160, 160))
        cv2.imwrite(out_file_name, resizedImage)
    else :
        extractedImg = faces[0].selection.extract(m_image)
        resizedImage = cv2.resize(extractedImg, (160, 160))
        cv2.imwrite(out_file_name, resizedImage)  

"""
    Pre process all the files in the folder
"""
def cut_images(folder_name, out_dir):
    
    if os.path.exists(out_dir):
        os.rmdir(out_dir)
    
    for file in os.listdir(folder_name):
        cut_image(os.path.join(folder_name, file) , out_dir)
import numpy as np
def predict(model_obj, img_file):
    img = cv2.imread(img_file)
    rep = model_obj.predict(np.array([img]))
    return rep[0]

from keras.models import load_model
def rep_all(model_file , in_dir, out_dir, result_file_path ):
    
    cut_images(in_dir , out_dir)
    loaded_model  = load_model(model_file)

    result = []
    file_ids = []
    for file_name in os.listdir(in_dir):
        processed_file = get_out_file_name(out_dir, file_name)
        result.append ( predict(load_model, processed_file))
        file_ids.append(file_name)
    result = np.column_stack([file_ids,result])
    np.savetxt( result_file_path , result, delimiter=',')
    
import sys
model_file = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3] 
result_file_path = sys.argv[4]

rep_all(model_file, in_dir, out_dir, result_file_path)



    
