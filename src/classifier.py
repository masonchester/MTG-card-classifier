
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

train_path = "images/train"
test_path = "images/test"
val_path = "images/validation"

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
                            str = class_name + ", " + subclass_name
                            labels.append(str)

    df = pd.DataFrame({
        "image paths":image_paths, 
        "labels":labels, 
          })
    return df
   
def gen_dataset(df_X):
    '''
    generates an X and y dataset consisting of image files 
    paths in X and one-hot encoded labels in y and zips them in a dataset object.
    '''
    data_X = df_X['image paths']
    data_X = tf.data.Dataset.from_tensor_slices(data_X)
    data_X = data_X.map(load_and_preprocess_image)

    data_y = df_X['labels'].str.get_dummies(', ')
    data_y = tf.data.Dataset.from_tensor_slices(data_y)

    dataset = tf.data.Dataset.zip((data_X, data_y))
    dataset = dataset.batch(64)
    return dataset

def load_and_preprocess_image(path):
    '''
    loads the images and resizes them to 299 x 299
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])
    return image

train_df = gen_dataframe_from_dir(train_path)
test_df = gen_dataframe_from_dir(test_path)
val_df = gen_dataframe_from_dir(val_path)

train_data = gen_dataset(train_df)
test_data  = gen_dataset(test_df)
val_data = gen_dataset(val_df)


base_model = tf.keras.applications.resnet_v2.ResNet152V2(
    include_top=False,
    weights='imagenet',
    input_shape=(299,299,3),
)

base_model.trainable = False

top_model = tf.keras.Sequential([
     tf.keras.layers.GlobalAveragePooling2D(),
     tf. keras.layers.Dense(1024, activation='relu'),  
     tf.keras.layers.Dense(12, activation='softmax')
])

model = tf.keras.models.Sequential([
    base_model,
    top_model
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_data, epochs=10, validation_data=val_data)