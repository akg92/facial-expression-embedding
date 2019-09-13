## clean the test and train file
import csv
from src.config.staticConfig  import StaticConfig
import os

def is_valid(urls, is_train):
    for url in urls:
        if not os.path.exists(StaticConfig.getImagePath(url), is_train):
            return False
    return True

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
                    common_eles = row[:16]
                    for i in range(16, len(row) , 2):
                        new_rows.append(list(common_eles))
                        new_rows[-1].append(row[i])
                        new_rows[-1].append(row[i+1])
                        total_new_rows += 1
                    csv_writer.writerows(new_rows)
    print('Total new rows {}, Total valid old rows'.format(total_new_rows, total_old_rows))


## create processed test and train files by removing non-existing files

                        
test_file_path = '../../data/faceexp-comparison-data-test-public.csv'
train_file_path = '../../data/faceexp-comparison-data-train-public.csv'

clean(train_file_path, True)
clean(test_file_path, True)


                
            

    

