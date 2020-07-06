import os
import json
import csv
import numpy as np
import pandas as pd
from optparse import OptionParser

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import generate_model

CLASS_INDICES = {'dog': 1, 'cat': 0, 'human': 2}  # ToDo: should be saved with training
OUTPUT_COLUMNS = {0: 1, 1: 0, 2: 2}  # to map CLASS_INDICES order


def main():
    parser = OptionParser(usage='usage: %prog -m model_path -o output_file [options] img1, img2, ...')
    parser.add_option('-m', '--model_path', dest='model_path', help='model to load')
    parser.add_option('-o', '--output_file', dest='output_file', help='path to csv output file')
    parser.add_option('-l', '--list_of_files', dest='list_of_files', help='list of files to predict')
    parser.add_option('-d', '--input_dir', dest='input_dir', help='dir with input files (jpg only)')
    (options, args) = parser.parse_args()
    if not options.model_path:
        parser.error('Required option -m.')
    if not options.output_file:
        parser.error('Required option -o.')

    # read config and model
    with open(os.path.splitext(options.model_path)[0] + '.conf') as json_read:
        config = json.load(json_read)
    model = generate_model(config)
    model.load_weights(options.model_path)

    data_frame = pd.DataFrame()

    # get files from arg
    for arg in args:
        data_frame = pd.concat([data_frame, pd.DataFrame({'filename': [arg]})])

    # get files from list
    if options.list_of_files:
        with open(options.list_of_files) as r:
            for file_name in r:
                data_frame = pd.concat([data_frame, pd.DataFrame({'filename': [file_name.strip()]})])

    # get files from dir
    if options.input_dir:
        for file_name in sorted(os.listdir(options.input_dir)):
            if os.path.splitext(file_name)[1].lower() == '.jpg':
                data_frame = pd.concat([data_frame, pd.DataFrame({'filename': [os.path.join(options.input_dir, file_name)]})])

    if len(data_frame) == 0:
        raise ValueError('No input files found.')

    # prepare data to process
    data_generator = ImageDataGenerator(rescale=1. / 255)
    data_flow = data_generator.flow_from_dataframe(
        data_frame,
        x_col='filename',
        batch_size=config['batch'],
        target_size=(config['height'], config['width']),
        class_mode=None,
        shuffle=False)

    # predict
    predictions = model.predict(data_flow)
    predictions = np.array(np.argmax(predictions, axis=-1))

    # write csv
    with open(options.output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['file_name']
        for c in CLASS_INDICES:
            header.append(c)
        writer.writerow(header)
        for f, p in zip(data_frame.values, predictions):
            pp = np.zeros((len(CLASS_INDICES), 1), int)
            pp[OUTPUT_COLUMNS[p]] = 1
            f = np.append(f, pp)
            writer.writerow(f)


if __name__ == '__main__':
    main()
