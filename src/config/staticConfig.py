
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
        if (StaticConfig._absCsvDir):
            return _absCsvDir
        else:
            return os.path.abspath(StaticConfig._relativeDataFolder)   
     
    
    @staticmethod
    def getTrainCSVPath():
        return os.path.join( StaticConfig.getCSVDir(), StaticConfig._trainCSVName)
    @staticmethod
    def getTestCSVPath():
        return os.path.join(StaticConfig.getCSVDir(), StaticConfig._testCSVName)
    
    @ staticmethod
    def getImageOutDir():

        if(StaticConfig._imgDownloadPath):
            return StaticConfig._imgDownloadPath
        else:
            return os.path.abspath(StaticConfig._relativeDataFolder)


    @staticmethod
    def _hashCode(s):
        h = 0
        for c in s:
            h = (31 * h + ord(c)) & 0xFFFFFFFF
        return str( abs(((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000))
    
    """
        Generate the image name from the url.
    """
    @staticmethod
    def getImagePath(url, train = True):
        last = url.split("/")[-1]
        fileName = "img_" + StaticConfig._hashCode(last) + "_" + StaticConfig._hashCode(url)+".jpg"
        imgDir = StaticConfig.getImageOutDir()
        return os.path.join(imgDir, "train", fileName) if train else os.path.join(imgDir, "test", fileName)
    @staticmethod
    def getImageProcessedPathPrefix(url, train):
        last = url.split("/")[-1]
        fileName = "img_" + StaticConfig._hashCode(last) + "_" + StaticConfig._hashCode(url)
        imgDir = StaticConfig.getImageOutDir()
        return os.path.join(imgDir, "train_processed", fileName) if train else os.path.join(imgDir, "test_processed", fileName)

    @staticmethod
    def getImageProcessedPath(url, train):
        last = url.split("/")[-1]
        fileName = "img_" + StaticConfig._hashCode(last) + "_" + StaticConfig._hashCode(url)+".jpg"
        imgDir = StaticConfig.getImageOutDir()
        return os.path.join(imgDir, "train_processed", fileName) if train else os.path.join(imgDir, "test_processed", fileName)

    @staticmethod 
    def getImageProcessedDir(train):
        imgDir = StaticConfig.getImageOutDir()
        return os.path.join(imgDir, "train_processed") if train else os.path.join(imgDir, "test_processed")







