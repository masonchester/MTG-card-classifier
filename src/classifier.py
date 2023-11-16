
import tensorflow as tf
import pandas as pd
import os

train_path = "/Users/masonchester/MTG-card-classifier/images/train"
test_path = "/Users/masonchester/MTG-card-classifier/images/test"
val_path = "/Users/masonchester/MTG-card-classifier/images/validation"

def gen_dataframe_from_dir (dir_path):
    '''
    generates a dataframe containing the image path and 
    labels from the directory structure.
    '''
    labels = list()
    image_paths = list()

    for class_name in os.listdir(dir_path):
        class_dir = os.path.join(dir_path, class_name)
        if os.path.isdir(class_dir):
            for subclass_name in os.listdir(class_dir):
                subclass_dir = os.path.join(class_dir,subclass_name)
                if os.path.isdir(subclass_dir):
                    for filename in os.listdir(subclass_dir):
                        file_dir = os.path.join(subclass_dir, filename)
                        if os.path.isfile(file_dir):
                            image_paths.append(file_dir)
                            str = class_name + "," + subclass_name
                            labels.append(str)

    df = pd.DataFrame({
        "image paths":image_paths, 
        "labels":labels, 
          })
    return df
   
def gen_dataset(df_X):
    '''
    generates image datasets from a provided dataframe 
    containing image filepaths in one columns and labels in the other.
    '''
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    data = datagen.flow_from_dataframe(
        df_X,
        x_col='image paths',
        y_col='labels',
        target_size=(244,244),
        batch_size=128,
        class_mode='categorical'
    )
    return data

train_df = gen_dataframe_from_dir(train_path)
test_df = gen_dataframe_from_dir(test_path)
val_df = gen_dataframe_from_dir(val_path)

train_data = gen_dataset(train_df)
test_data = gen_dataset(test_df)
val_data = gen_dataset(val_df)

base_model = tf.keras.applications.resnet_v2.ResNet152V2(
    include_top=False,
    weight='imagenet',
    input_shape=(244,244,3),
)

base_model.trainable = False

top_model = tf.keras.Sequential([
     tf. keras.layers.Dense(1024, activation='relu'),  
     tf.keras.layers.Dense(36, activation='softmax')
])

model = tf.keras.models.Sequential([
    base_model,
    top_model
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=val_data)