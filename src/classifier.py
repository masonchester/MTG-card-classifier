
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
             for filename in os.listdir(class_dir):
                file_dir = os.path.join(class_dir, filename)
                if os.path.isfile(file_dir):
                    image_paths.append(file_dir)
                    str = class_name
                    labels.append(str)

    df = pd.DataFrame({
        "image paths":image_paths, 
        "labels":labels, 
          })

    print(df.sample(5))
    return df
   
def gen_dataset(df_X):
    '''
    generates an X and y dataset consisting of image files 
    paths in X and one-hot encoded labels in y and zips them in a dataset object.
    '''
    data_X = df_X['image paths']
    data_y = df_X['labels'].str.get_dummies(',')

    dataset = tf.data.Dataset.from_tensor_slices((data_X, data_y))
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.batch(64)

    return dataset

def load_and_preprocess_image(image, label):
    '''
    loads the images and resizes them to 299 x 299
    '''
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])
    image = tf.keras.applications.resnet_v2.preprocess_input(image)

    return image, label

train_df = gen_dataframe_from_dir(train_path)
test_df = gen_dataframe_from_dir(test_path)
val_df = gen_dataframe_from_dir(val_path)

train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)
val_df = val_df.sample(frac=1)

train_data = gen_dataset(train_df)
test_data  = gen_dataset(test_df)
val_data = gen_dataset(val_df)


base_model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3),
)

base_model.trainable = False

top_model = tf.keras.Sequential([
     tf.keras.layers.GlobalAveragePooling2D(),
     tf. keras.layers.Dense(1024, activation='relu'),  
     tf.keras.layers.Dense(6, activation='softmax')
])

model = tf.keras.models.Sequential([
    base_model,
    top_model
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_data, epochs=10, validation_data=val_data)