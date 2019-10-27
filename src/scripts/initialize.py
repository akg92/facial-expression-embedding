import sys
sys.path.append('..')
from data.dataRep import FCEXPDataSet
from config.staticConfig import StaticConfig 
from src.limit import limitUsage
### Command line options should be added in future

limitUsage("7")
## download images from the url
trainEntries = FCEXPDataSet(StaticConfig.getTrainCSVPath())
#trainEntries.downloadImages()
trainEntries.cutImages()
del trainEntries
testEntries = FCEXPDataSet(StaticConfig.getTestCSVPath(), "test")
testEntries.cutImages()
#testEntries.downloadImages()
del testEntries



