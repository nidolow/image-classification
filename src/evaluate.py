import os
import json
import numpy as np
from optparse import OptionParser

from train import get_data_frames, generate_data_flow, generate_model

DATA_PATH = 'data/train/'


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model_path', dest='model_path', help='model to load')
    (options, args) = parser.parse_args()
    with open(os.path.splitext(options.model_path)[0]+'.conf') as json_read:
        config = json.load(json_read)

    _, test_df = get_data_frames(DATA_PATH)
    test_data = generate_data_flow(test_df, config, DATA_PATH, augment=False, shuffle=False)

    model = generate_model(config)
    model.load_weights(options.model_path)

    predictions = model.predict(test_data, steps=np.ceil(len(test_df)/config['batch']))
    predictions = np.array(np.argmax(predictions, axis=-1))

    test_df['labels'] = test_df['category'].replace(test_data.class_indices)
    labels = np.array(test_df['labels'])
    labels_len = len(np.array(test_df['labels']))

    print('Total error:', sum(predictions != labels)/labels_len)
    for k in test_data.class_indices:
        v = test_data.class_indices[k]
        print(k)
        print(sum((predictions != labels) & (labels == v))/sum((labels == v)))


if __name__ == "__main__":
    main()
