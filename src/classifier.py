
import tensorflow as tf
import pandas as pd
import os
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
                            str = class_name + "," + subclass_name
                            labels.append(str)

    df = pd.DataFrame({
        "image paths":image_paths, 
        "labels":labels, 
          })
    return df
   
def gen_dataset(df_X):
    # TO-DO: Implement a way to encode the labels from the dataframes
    return data

train_df = gen_dataframe_from_dir(train_path)
test_df = gen_dataframe_from_dir(test_path)
val_df = gen_dataframe_from_dir(val_path)

train_data = gen_dataset(train_df)
test_data = gen_dataset(test_df)
val_data = gen_dataset(val_df)

def plot_images_from_generator(generator, num_images=5):
    # Get a batch of images and labels from the generator
    for i, (images, labels) in enumerate(generator):
        # Set up the grid of subplots
        num_rows = num_images // 5 + int(num_images % 5 != 0)
        plt.figure(figsize=(20, 4 * num_rows))
        
        # Only take the number of images you want to plot
        for j in range(num_images):
            plt.subplot(num_rows, 5, j+1)
            plt.imshow(images[j])
         
            plt.axis('off')
        
        # Break the loop after the first batch to prevent infinite loops
        break

    plt.tight_layout()
    plt.show()

# Plot images from the training data
plot_images_from_generator(train_data, num_images=10)

base_model = tf.keras.applications.resnet_v2.ResNet152V2(
    include_top=False,
    weights='imagenet',
    input_shape=(299,299,3),
)

base_model.trainable = False

top_model = tf.keras.Sequential([
     tf.keras.layers.Flatten(),
     tf. keras.layers.Dense(1024, activation='relu'),  
     tf.keras.layers.Dense(36, activation='softmax')
])

model = tf.keras.models.Sequential([
    base_model,
    top_model
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=val_data)