#!/usr/bin/env python
# coding: utf-8


import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

DATA_PATH = 'data/train/'
OUTPUT_DIR = 'models/'
CONF = {
    'batch': 128,
    'max_epochs': 50,
    'height': 128,
    'width': 128,
    'learning_rate': 0.0001,
    'early_stop': True,
    'batch_norm': False,
    'dropout': True,
    'features': {}
}


def limit_gpu_mem(max_gpu_mem=1536):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:  # Better GPU works fine with no restrictions
        # Restrict TensorFlow to only allocate limited amount of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_gpu_mem)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def get_data_frames(data_path=DATA_PATH):
    df = pd.DataFrame()
    for category in os.listdir(data_path):
        print('Loading category:', category)
        filenames = [os.path.join(category, f) for f in os.listdir(os.path.join(data_path, category))]
        df = pd.concat([df,
                        pd.DataFrame({'filename': filenames,
                                      'category': category})])
    train_df, validation_df = train_test_split(df, test_size=0.10, random_state=29)
    return train_df, validation_df


def generate_data_flow(data_frame, data_path=DATA_PATH):
    data_generator = ImageDataGenerator(rescale=1. / 255)
    data_flow = data_generator.flow_from_dataframe(
        data_frame,
        data_path,
        x_col='filename',
        y_col='category',
        batch_size=CONF['batch'],
        target_size=(CONF['height'], CONF['width']),
        class_mode='categorical')
    return data_flow


def generate_model():
    model = Sequential()

    model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(CONF['height'], CONF['width'], 3)))
    if CONF['batch_norm']: model.add(BatchNormalization())
    model.add(MaxPooling2D())
    if CONF['dropout']: model.add(Dropout(0.25))

    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    if CONF['batch_norm']: model.add(BatchNormalization())
    model.add(MaxPooling2D())
    if CONF['dropout']: model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    if CONF['batch_norm']: model.add(BatchNormalization())
    model.add(MaxPooling2D())
    if CONF['dropout']: model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    if CONF['dropout']: model.add(Dropout(0.25))
    if CONF['batch_norm']: model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(lr=CONF['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    CONF['model'] = json.loads(model.to_json())
    return model


def train(model, train_data, validation_data, steps_per_epoch, validation_steps):
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=CONF['max_epochs'],
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=[early_stop])
    return history


def serialize(model, history, output_dir=OUTPUT_DIR):
    name_hash = str(hashlib.md5(json.dumps(CONF, sort_keys=True).encode("utf-8")).hexdigest()[0:7])

    model.save_weights(os.path.join(output_dir, 'model-' + name_hash + '.mdl'))
    with open(os.path.join(output_dir, 'model-' + name_hash + '.history'), 'w') as w:
        pd.DataFrame(history.history).to_json(w)
    with open(os.path.join(output_dir, 'model-' + name_hash + '.conf'), 'w') as w:
        json.dump(CONF, w)


def main():
    train_df, validation_df = get_data_frames(DATA_PATH)
    train_data = generate_data_flow(train_df, DATA_PATH)
    validation_data = generate_data_flow(validation_df, DATA_PATH)
    model = generate_model()
    history = train(model,
                    train_data,
                    validation_data,
                    len(train_df)//CONF['batch'],
                    len(validation_df)//CONF['batch'])
    serialize(model, history, OUTPUT_DIR)


if __name__ == "__main__":
    main()
