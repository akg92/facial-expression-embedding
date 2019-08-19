import urllib.request
import os
import threading
import csv
from src.config.staticConfig import StaticConfig as StaticConfig
class Downloader():

    @staticmethod
    def download(url):
        destPath = StaticConfig.getImagePath(url, train=True)
        if( os.path.exists(destPath)):
            return True
        try:
            urllib.request.urlretrieve(url , destPath)
        except :
            return False
        return True
    


