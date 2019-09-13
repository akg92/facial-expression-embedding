## clean the test and train file
import csv
from src.config.staticConfig  import StaticConfig
from src.data.dataRep import Image
import os
from src.limit import limitUsage
## to avoid accidental  gpu usage
limitUsage("5,6")
def is_valid(urls, is_train):
    for url in urls:
        if not os.path.exists(StaticConfig.getImagePath(url, is_train)):
            return False
    return True
all_valid_entries = {}
def get_key(is_train, row, index):
    return '_'.join(row[index:index+5])+'_'+str(is_train)

def get_processed_file_name( is_train, row, index):
    key = get_key(is_train, row, index)
    if( key in all_valid_entries):
        return all_valid_entries[key]
    start_index = index*5
    img = Image(row[start_index], row[start_index+1], row[start_index+2], row[start_index+3], row[start_index+4], False)
    val = img.getProcessedFilePath(is_train)
    all_valid_entries[key] = val
    return val

    

def clean(file_name, is_train):
    total_new_rows = 0
    total_old_rows = 0
    img_indexes = [0, 5, 10]
    fname = os.path.basename(file_name)
    processed_file = os.path.join( os.path.dirname(file_name),'processed_'+fname) 
    with open(file_name) as f:
        with open(processed_file, 'w') as wfile:
            csv_reader = csv.reader(f)
            csv_writer = csv.writer(wfile)
            
            for row in csv_reader:
                all_file_exist = is_valid( [ row[i] for i in img_indexes], is_train)
                ## create multi rows
                if( all_file_exist):
                    total_old_rows += 1
                    new_rows = []
                    common_eles = list(row[:16])
                    common_eles.append(get_processed_file_name(is_train , row, 0))
                    common_eles.append(get_processed_file_name(is_train , row, 1))
                    common_eles.append(get_processed_file_name(is_train , row, 2))

                    for i in range(16, len(row) , 2):
                        new_rows.append(list(common_eles))
                        new_rows[-1].append(row[i])
                        new_rows[-1].append(row[i+1])
                        total_new_rows += 1
                    csv_writer.writerows(new_rows)
    print('Total new rows {}, Total valid old rows {}'.format(total_new_rows, total_old_rows))


## create processed test and train files by removing non-existing files

                        
test_file_path = '../../data/faceexp-comparison-data-test-public.csv'
train_file_path = '../../data/faceexp-comparison-data-train-public.csv'

clean(train_file_path, True)
clean(test_file_path, True)


                
            

    

