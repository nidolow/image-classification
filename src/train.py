#!/usr/bin/env python
# coding: utf-8


import os
import json
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from optparse import OptionParser

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

DATA_PATH = 'data/train/'
OUTPUT_DIR = 'models/'


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


def get_data_frames(data_path):
    df = pd.DataFrame()
    for category in os.listdir(data_path):
        print('Loading category:', category)
        filenames = [os.path.join(category, f) for f in os.listdir(os.path.join(data_path, category))]
        df = pd.concat([df,
                        pd.DataFrame({'filename': filenames,
                                      'category': category})])
    train_df, validation_df = train_test_split(df, test_size=0.10, random_state=29)
    return train_df, validation_df


def generate_data_flow(data_frame, conf, data_path, augment=False):
    if augment:
        data_generator = ImageDataGenerator(rescale=1. / 255,
                                            horizontal_flip=True,
                                            rotation_range=30,
                                            zoom_range=0.2,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2)
    else:
        data_generator = ImageDataGenerator(rescale=1. / 255)
    data_flow = data_generator.flow_from_dataframe(
        data_frame,
        data_path,
        x_col='filename',
        y_col='category',
        batch_size=conf['batch'],
        target_size=(conf['height'], conf['width']),
        class_mode='categorical')
    return data_flow


def generate_model(conf):
    model = Sequential()

    model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(conf['height'], conf['width'], 3)))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(MaxPooling2D())
    if conf['dropout']: model.add(Dropout(0.25))

    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(MaxPooling2D())
    if conf['dropout']: model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(MaxPooling2D())
    if conf['dropout']: model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    if conf['dropout']: model.add(Dropout(0.25))
    if conf['batch_norm']: model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(lr=conf['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def train(model, train_data, validation_data, steps_per_epoch, validation_steps, epochs, patience):
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)
    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=[early_stop])
    return history


def serialize(model, history, conf, output_dir):
    conf['model'] = json.loads(model.to_json())
    name_hash = str(hashlib.md5(json.dumps(conf, sort_keys=True).encode("utf-8")).hexdigest()[0:7])

    model.save_weights(os.path.join(output_dir, 'model-' + name_hash + '.mdl'))
    with open(os.path.join(output_dir, 'model-' + name_hash + '.history'), 'w') as w:
        pd.DataFrame(history.history).to_json(w)
    with open(os.path.join(output_dir, 'model-' + name_hash + '.conf'), 'w') as w:
        json.dump(conf, w)


def main():
    config = {
        'dropout': True
    }
    parser = OptionParser()
    parser.add_option('-b', '--batch', dest='batch', default=128, type='int', help='batch size')
    parser.add_option('--width', dest='width', default=128, type='int', help='image width')
    parser.add_option('--height', dest='height', default=128, type='int', help='image height')
    parser.add_option('-l', '--learning_rate', dest='learning_rate', default=0.0001, type='float', help='learning rate')
    parser.add_option('--max_epochs', dest='max_epochs', default=50, type='int', help='max number of epochs')
    parser.add_option('-e', '--early_stop', dest='early_stop', default=5, type='int', help='early stopping')
    parser.add_option('-n', '--batch_norm', action='store_true', dest='batch_norm', default=False,
                      help='add batch normalization')
    parser.add_option('-a', '--data_augment', action='store_true', dest='data_augment', default=False,
                      help='add batch normalization')
    (options, args) = parser.parse_args()
    config.update(vars(options))

    train_df, validation_df = get_data_frames(DATA_PATH)
    train_data = generate_data_flow(train_df, config, DATA_PATH, config['data_augment'])
    validation_data = generate_data_flow(validation_df, config, DATA_PATH)
    model = generate_model(config)
    print(config)
    history = train(model,
                    train_data,
                    validation_data,
                    steps_per_epoch=len(train_df)//config['batch'],
                    validation_steps=len(validation_df)//config['batch'],
                    epochs=config['max_epochs'],
                    patience=config['early_stop'])
    serialize(model, history, config, OUTPUT_DIR)


if __name__ == "__main__":
    main()
