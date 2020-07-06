#!/usr/bin/env python
# coding: utf-8


import os
import json
import hashlib
import pandas as pd
from optparse import OptionParser
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models import generate_model


def get_train_data_frames(data_path):
    """
    Loads training files paths and pre-splits them into sets.

    Args:
        data_path (str): training data directory

    Returns:
        train_df (DataFrame): training set 80%
        validation_df (DataFrame): validation set 10%
        train_df (DataFrame): test set 10%
    """

    df = pd.DataFrame()
    for category in sorted(os.listdir(data_path)):
        print('Loading category:', category)
        filenames = [os.path.join(category, f) for f in sorted(os.listdir(os.path.join(data_path, category)))]
        df = pd.concat([df,
                        pd.DataFrame({'filename': filenames,
                                      'category': category})])
    train_df, test_validation_df = train_test_split(df, test_size=0.20, random_state=29)
    test_df, validation_df = train_test_split(test_validation_df, test_size=0.50, random_state=31)
    return train_df, validation_df, test_df


def generate_data_flow(data_frame, conf, data_path, shuffle=True):
    """
    Generates batches of tensor data image from dataframe.

    Args:
        data_frame (DataFrame): input data
        conf (dict): training configuration parameters
        data_path (str): training data directory
        shuffle (bool): if True shuffles order of data files

    Returns:
        data_flow (DirectoryIterator): iterator capable of reading images from disk
    """

    if conf['data_augment']:
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
        class_mode='categorical',
        shuffle=shuffle)
    return data_flow


def train(model, train_data, validation_data, steps_per_epoch, validation_steps, epochs, learning_rate, patience):
    """
        Trains model for given number of epochs or until early stopping condition
        is met.

        Args:
            model (Sequential): object with model structure to train
            train_data (DirectoryIterator): iterator for training data
            validation_data (DirectoryIterator): iterator for validation data
            steps_per_epoch (int): number of training steps per epoch
            validation_steps (int): number of validation steps per epoch
            epochs (int): maximum number of epochs to train
            learning_rate (int): starting learning rate
            patience (int): number of epochs with no improvement after which training
                will be stopped

        Returns:
            history (History): object with record of training loss and accuracy values
        """

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1)

    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_loss',
        patience=2,
        verbose=1,
        factor=0.5,
        min_lr=0.00001)

    history = model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=[early_stop, learning_rate_reduction])
    return history


def serialize(model, history, conf, output_dir):
    """
    Saves model weighs, training configuration and training history.

    Args:
        model (Sequential): object with trained model structure
        history (History): object with record of training loss and accuracy values
        conf (dict): model and training configuration parameters
        output_dir (str): directory to save files with training results
    """

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
    parser = OptionParser(usage='usage: %prog --arch arg [options]')
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
    parser.add_option('--arch', dest='arch', help='model architecture (vgg_v1|vgg_v2|baseline)')
    parser.add_option('--data_path', dest='data_path', default='data/train/',
                      help='path to training files directory')
    parser.add_option('--output_dir', dest='output_dir', default='models/',
                      help='directory to save files with training results')
    (options, args) = parser.parse_args()
    if options.arch is None:
        parser.error('Required option --arch (vgg_v1|vgg_v2|baseline).')
    config.update(vars(options))

    train_df, validation_df, _ = get_train_data_frames(options.data_path)
    train_data = generate_data_flow(train_df, config, options.data_path)
    validation_data = generate_data_flow(validation_df, config, options.data_path)

    model = generate_model(config)
    model.summary()
    print(config)

    history = train(model,
                    train_data,
                    validation_data,
                    steps_per_epoch=len(train_df) // config['batch'],
                    validation_steps=len(validation_df) // config['batch'],
                    epochs=config['max_epochs'],
                    learning_rate=config['learning_rate'],
                    patience=config['early_stop'])
    serialize(model, history, config, options.output_dir)


if __name__ == '__main__':
    main()
