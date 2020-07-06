import os
import json
import numpy as np
from optparse import OptionParser

from models import generate_model
from train import get_train_data_frames, generate_data_flow


def evaluate(model_path, data_path):
    """
        Evaluates given model with test set separated from all training data.
        Gives details about accuracy for every class. Outputs results od screen.

        Args:
            model_path (str): path to model file
            data_path (str): training data directory
    """

    with open(os.path.splitext(model_path)[0] + '.conf') as json_read:
        config = json.load(json_read)

    _, _, test_df = get_train_data_frames(data_path)
    test_data = generate_data_flow(test_df, config, data_path, shuffle=False)

    model = generate_model(config)
    model.load_weights(model_path)

    predictions = model.predict(test_data, steps=np.ceil(len(test_df) / config['batch']))
    predictions = np.array(np.argmax(predictions, axis=-1))

    test_df['labels'] = test_df['category'].replace(test_data.class_indices)
    labels = np.array(test_df['labels'])
    labels_len = len(np.array(test_df['labels']))

    print('\nModel:', model_path)
    print('Total accuracy:', sum(predictions == labels) / labels_len)
    for k in test_data.class_indices:
        v = test_data.class_indices[k]
        print(k, 'accuracy')
        print(sum((predictions == labels) & (labels == v)) / sum((labels == v)))
    print('')


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model_path', dest='model_path', help='model to load')
    parser.add_option('-d', '--models_dir', dest='models_dir', help='path to directory with models')
    parser.add_option('--data_path', dest='data_path', default='data/train/',
                      help='path to training files directory')
    (options, args) = parser.parse_args()
    if not options.model_path and not options.models_dir:
        parser.error('Requires option -m or -d.')

    if options.model_path:
        evaluate(options.model_path, options.data_path)
    if options.models_dir:
        for file_path in os.listdir(options.models_dir):
            root, ext = os.path.splitext(file_path)
            if ext == '.index' and os.path.splitext(root)[-1] == '.mdl':
                evaluate(os.path.join(options.models_dir, root), options.data_path)


if __name__ == '__main__':
    main()
