import tensorflow as tf
import pandas as pd
import os

train_path = "/Users/masonchester/MTG-card-classifier/images/train"
test_path = "/Users/masonchester/MTG-card-classifier/images/test"
val_path = "/Users/masonchester/MTG-card-classifier/images/validation"

def dataframe_from_dir (dir_path):
    '''
    generates a dataframe containing the image path, class names and subclass names.
    '''
    
    class_labels = list()
    subclass_labels = list()
    image_paths = list()

    for class_name in os.listdir(dir_path):
        class_dir = os.path.join(dir_path, class_name)
        if os.path.isdir(class_dir):
            for subclass in os.listdir(class_dir):
                subclass_dir = os.path.join(class_dir,subclass)
                if os.path.isdir(subclass_dir):
                    for filename in os.listdir(subclass_dir):
                        file_dir = os.path.join(subclass_dir, filename)
                        if os.path.isfile(file_dir):
                            image_paths.append(file_dir)
                            class_labels.append(class_name)
                            subclass_labels.append(subclass)

    df = pd.DataFrame({
        "image paths":image_paths, 
        "class_labels":class_labels, 
        "subclass_labels":subclass_labels
          })
    return df

train_df = dataframe_from_dir(train_path)

print(train_df.sample(5))
