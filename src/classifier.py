import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score

train_path = "images/train"
test_path = "images/test"
val_path = "images/validation"

def gen_dataframe_from_dir(dir_path):
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
        "image paths": image_paths,
        "labels": labels,
    })

    print(df.sample(5))
    return df

def gen_dataset(df_X, augment=False):
    data_X = df_X['image paths']
    data_y = df_X['labels']

    mlb = LabelBinarizer()
    data_y = mlb.fit_transform(data_y)
    dataset = tf.data.Dataset.from_tensor_slices((data_X, data_y))
    dataset = dataset.map(load_and_preprocess_image)
    dataset = dataset.batch(64)

    return dataset

def load_and_preprocess_image(image, label):
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

train_data = gen_dataset(train_df, augment=True)
test_data = gen_dataset(test_df)
val_data = gen_dataset(val_df)

base_model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(299, 299, 3),
)

base_model.trainable = True
fine_tune_at = 100  # Adjust this value based on the layer you want to fine-tune from
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

top_model = tf.keras.Sequential([
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dense(6, activation='softmax')
])

model = tf.keras.models.Sequential([
    base_model,
    top_model
])

# Learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Training with data augmentation and early stopping
history = model.fit(x=train_data, epochs=10, validation_data=val_data, callbacks=[early_stopping])

# Extracting metrics
training_accuracy = sum(history.history['accuracy']) / 10
validation_accuracy = sum(history.history['val_accuracy']) / 10
loss = sum(history.history['loss']) / 10
validation_loss = sum(history.history['val_loss']) / 10

print(f'Training Accuracy: {training_accuracy * 100:.2f}%')
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')
print(f'Training Loss: {loss:.4f}')
print(f'Validation Loss: {validation_loss:.4f}')

# Evaluation metrics for precision and recall
predictions = model.predict(test_data)

true_labels = list()
for _,labels in test_data:
    for label in labels:
        true_labels.append(np.array(label))

mlb = LabelBinarizer()
mlb.fit(test_data)
predictions = mlb.inverse_transform(predictions)
true_labels = mlb.inverse_transform(true_labels)
mlcm = multilabel_confusion_matrix(y_true=true_labels, y_pred=predictions)

# Calculate precision, recall, and F1-score
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
