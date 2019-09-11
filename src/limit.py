import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

def limitUsage(deviceList):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = deviceList #only the gpu 0 is allowed
    set_session(tf.Session(config=config))