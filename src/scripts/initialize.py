import sys
sys.path.append('..')
from data.dataRep import FCEXPDataSet
from config.staticConfig import StaticConfig 
### Command line options should be added in future


## download images from the url
trainEntries = FCEXPDataSet(StaticConfig.getTrainCSVPath())
trainEntries.downloadImages()
del trainEntries
testEntries = FCEXPDataSet(StaticConfig.getTestCSVPath())
testEntries.downloadImages()
del testEntries


