import urllib
import os
import wget
import threading
import csv
from config.staticConfig import StaticConfig as StaticConfig
from threading import Lock
lock = Lock()
class Downloader():
    errors = []
    writing = False
    @staticmethod
    def error_log(log):
        while(Downloader.writing):
            pass
        Downloader.errors.append(log)
        if( len(log) > 20):
            lock.acquire()
            Downloader.writing = True
            to_write = Downloader.errors
            Downloader.errors = []
            Downloader.writing = False
            lock.release()
            if(len(log)> 20):
                with open('error.txt', 'w+') as f:
                    f.write('\n'.join(to_write))



    @staticmethod
    def download(train,url):
        destPath = StaticConfig.getImagePath(url, train)
        #print(destPath)
        folder =  "train" if train else "test"
        #print(url)
        try:
            os.mkdir( os.path.join(StaticConfig.getImageOutDir(),folder))
        except:
            pass 
        if( os.path.exists(destPath)):
            return True
        try:
            ##urlo = urllib.request.url
            wget.download(url,  destPath)
            
        except Exception as e :
            print(url)
            Downloader.error_log(url)
            return False
        return True
    


