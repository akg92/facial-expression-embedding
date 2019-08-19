
import os
import urllib
class StaticConfig:

    _relativeDataFolder = '../../data'
    _imgDownloadPath = None 
    _csvRelativePath = None 
    _absCsvDir = None 
    _testCSVName = 'faceexp-comparison-data-test-public.csv'
    _trainCSVName = 'faceexp-comparison-data-train-public.csv'

    @staticmethod
    def getCSVDir():
        if (_absCsvDir):
            return _absCsvDir
        else:
            return os.path.abspath(_relativeDataFolder)   
     
    
    @staticmethod
    def getTrainCSVPath():
        return os.path.join(getCSVDir(), _trainCSVName)
    @staticmethod
    def getTestCSVPath():
        return os.path.join(getCSVDir(), _testCSVName)
    
    @ staticmethod
    def getImageOutDir():

        if(_imgDownloadPath):
            return _imgDownloadPath
        else:
            return os.path.abspath(_relativeDataFolder)

    @staticmethod
    def _hashCode(s):
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    
    """
        Generate the image name from the url.
    """
    @staticmethod
    def getImagePath(url, train = True):
        last = url.split("/")[-1]
        fileName = "img_" + _hashCode(last) + "_" + _hashCode(url)
        imgDir = getImageOutDir()
        return os.path.join(imgDir, "train", fileName) if train else os.path.join(imgDir, "test", fileName)



