
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from .preprocess import Downloader
import os
import cv2
from mira.core import Image as miraImage
from mira import detectors
from src.config.staticConfig import StaticConfig 
class Image():

    def __init__(self, url, topLeftCol, bottomRightCol, topLeftRow, bottomRightRow ):
        self.url = url
        self.url = self.url.replace("\"","")
        self.topLeftCol = float(topLeftCol)
        self.bottomRightCol = float(bottomRightCol)
        self.topLeftRow = float(topLeftRow)
        self.bottomRightRow = float(bottomRightRow)
        self.width = -1
    
    def  resizeImage(self, isTrain):
        originalFileName = StaticConfig.getImagePath(self.url, isTrain)
        if not os.path.exists(originalFileName):
            return False

        processedFileNamePrefix = StaticConfig.getImageProcessedPath(self.url, isTrain)
        npArray = cv2.imread(originalFileName)
        width = npArray.shape[1]
        height = npArray.shape[0]
        tleftcol = int( self.topLeftCol*width)
        brightcol = int(self.bottomRightCol*width)
        topleftrow = int(self.topLeftRow*height)
        brighttrow = int(self.bottomRightRow*height)

        processdFileName = processedFileNamePrefix + "{}_{}_{}_{}.jpg".format(tleftcol, brightcol, topleftrow, brighttrow)
        tempFileName = processedFileNamePrefix + "{}_{}_{}_{}_temp.jpg".format(tleftcol, brightcol, topleftrow, brighttrow)
        ## file exist
        if( os.path.exists(processdFileName)):
            return True
        cutIme = npArray[tleftcol:brightcol, topleftrow:brighttrow]
        cv2.imwrite(tempFileName, cutIme)
        ## do preprocessing
        mImage = miraImage.read(tempFileName)
        detector = detectors.MTCNN()
        faces = detector.detect(mImage)
        if( not faces or not faces[0]):
            print('face_not_found for {}'.format(originalFileName))
            resizedImage = cv2.resize(cutIme, (160, 160))
            cv2.imwrite(processdFileName, resizedImage) 
        
        else :
            extractedImg = faces[0].selection.extract(mImage)
            resizedImage = cv2.resize(extractedImg, (160, 160))
            cv2.imwrite(processdFileName, resizedImage) 
        ## clear temp file
        os.remove(tempFileName)
        





        

        



        cv2.imwrite(npArray, )




class Rating():
    def __init__(self, raterId, rating):
        self.raterId = int(raterId)
        self.rating = int(rating)
        
class Entry():

    HEADER = None

    

    def __init__(self, row):
        self.type = None
        self.images = []
        self.ratings = []
        fields = row.split(",")

        for i in range(3):
            col = i * 5
            self.images.append( Image(fields[col], fields[col+1], fields[col+2], fields[col+3], fields[col+4]))
        
        self.type = fields[15]
        for i in range(16, len(fields) , 2):
            self.ratings.append(Rating( fields[i], fields[i+1]))
        
class FCEXPDataSet():

    def __init__(self, fileName, type = "train"):
        self.type = type
        self.fileName = fileName
        self.entries = []
        self.loadData()
    """
    Load csv data to object representation
    """
    def loadData(self):

        with open(self.fileName,'r') as f:

            for line in f:
                self.entries.append(Entry(line))


    

    def downloadImages(self,nThread = 10):
        pool = ThreadPoolExecutor(nThread)
        Downloader.error_set = set()
        for entry in self.entries:
            #Downloader.download(entry.url)
            for e in entry.images:
                ##pool.submit(Downloader.download, ( self.type == 'train', e.url))
                Downloader.download( self.type == 'train', e.url)

        with open('total_error.txt','w+') as f:
            f.write("Total error {} {} \n".format(self.type,len(Downloader.error_set)))

    """
        Cut image
    """
    def cutImages(self):
        isTrain = self.type == 'train'
        ## create dir
        ## each image may have multiple 
        if(isTrain):
            if ( not os.path.exists(StaticConfig.getImageProcessedDir(isTrain))):
                os.mkdir(StaticConfig.getImageProcessedDir(isTrain))
        
        if( not isTrain ):
            if ( not os.path.exists(StaticConfig.getImageProcessedDir(isTrain))):
                os.mkdir(StaticConfig.getImageProcessedDir(isTrain))
        
        for entry in self.entries:
            for img in entry.images:
                img.resizeImage(isTrain)
        


        
        
    




        
