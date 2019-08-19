import urllib
import os
import wget
import threading
import csv
from config.staticConfig import StaticConfig as StaticConfig
class Downloader():

    @staticmethod
    def download(train,url):
        destPath = StaticConfig.getImagePath(url, train)
        #print(destPath)
        folder =  "train" if train else "test"
        print(url)
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
            print(e)
            return False
        return True
    


