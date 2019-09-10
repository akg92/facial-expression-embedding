import sys
sys.path.append('..')
from data.dataRep import FCEXPDataSet
from config.staticConfig import StaticConfig 
### Command line options should be added in future


## download images from the url
trainEntries = FCEXPDataSet(StaticConfig.getTrainCSVPath())
trainEntries.downloadImages()
trainEntries.cutImages()
del trainEntries
testEntries = FCEXPDataSet(StaticConfig.getTestCSVPath(), "test")
testEntries.cutImages()
testEntries.downloadImages()
del testEntries



