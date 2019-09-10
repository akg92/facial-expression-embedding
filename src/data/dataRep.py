
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from .preprocess import Downloader
class Image():

    def __init__(self, url, topLeftCol, bottomRightCol, topLeftRow, bottomRightRow ):
        self.url = url
        self.url = self.url.replace("\"","")
        self.topLeftCol = float(topLeftCol)
        self.bottomRightCol = float(bottomRightCol)
        self.topLeftRow = float(topLeftRow)
        self.bottomRightRow = float(bottomRightRow)
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

        
        
    




        
