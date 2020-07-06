from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, ReLU


def add_conv(model, filters, dropout, batch_norm):
    """
    Adds to model convolution layer with optional dropout and batch normalization.

    Args:
        model (Sequential): object with model structure
        filters (int): number of filters
        dropout (bool): if True adds dropout
        batch_norm (boot): if True adds batch normalization
    """

    model.add(Conv2D(filters, 3, padding='same', activation=None))
    if batch_norm: model.add(BatchNormalization())
    model.add(ReLU())
    if dropout: model.add(Dropout(0.25))


def generate_model(conf):
    """
        Initializes model structure.

        Args:
            conf (dict): model configuration parameters

        Returns:
            model (Sequential): object with model structure

        Raises:
            KeyError: if there is not 'arch' key in configuration dict
            ValueError: if 'arch' parameter does not match known model structure
    """

    if 'arch' not in conf:
        raise KeyError('Missing "arch" in config.')
    if conf['arch'] == 'vgg_v1':
        return get_vgg_v1(conf)
    if conf['arch'] == 'vgg_v2':
        return get_vgg_v2(conf)
    if conf['arch'] == 'baseline':
        return get_baseline(conf)
    raise ValueError('Unknown value for "arch" in config: '+conf['arch'])


def get_vgg_v1(conf):
    """
        Generates object with predefined VGG model structure.

        Args:
            conf (dict): model configuration parameters

        Returns:
            model (Sequential): object with model structure
    """

    model = Sequential()

    model.add(Conv2D(32, 3, padding='same', activation=None, input_shape=(conf['height'], conf['width'], 3)))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(ReLU())
    if conf['dropout']: model.add(Dropout(0.25))
    add_conv(model, 32, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 64, conf['dropout'], conf['batch_norm'])
    add_conv(model, 64, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 128, conf['dropout'], conf['batch_norm'])
    add_conv(model, 128, conf['dropout'], conf['batch_norm'])
    add_conv(model, 128, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 256, conf['dropout'], conf['batch_norm'])
    add_conv(model, 256, conf['dropout'], conf['batch_norm'])
    add_conv(model, 256, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation=None))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(ReLU())
    if conf['dropout']: model.add(Dropout(0.25))

    model.add(Dense(512, activation=None))
    if conf['dropout']: model.add(Dropout(0.25))
    model.add(ReLU())
    if conf['batch_norm']: model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    return model


def get_vgg_v2(conf):
    """
        Generates object with predefined bigger VGG model structure.

        Args:
            conf (dict): model configuration parameters

        Returns:
            model (Sequential): object with model structure
    """

    model = Sequential()

    model.add(Conv2D(64, 3, padding='same', activation=None, input_shape=(conf['height'], conf['width'], 3)))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(ReLU())
    if conf['dropout']: model.add(Dropout(0.25))
    add_conv(model, 64, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 128, conf['dropout'], conf['batch_norm'])
    add_conv(model, 128, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 256, conf['dropout'], conf['batch_norm'])
    add_conv(model, 256, conf['dropout'], conf['batch_norm'])
    add_conv(model, 256, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    add_conv(model, 512, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation=None))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(ReLU())
    if conf['dropout']: model.add(Dropout(0.25))

    model.add(Dense(512, activation=None))
    if conf['dropout']: model.add(Dropout(0.25))
    model.add(ReLU())
    if conf['batch_norm']: model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    return model


def get_baseline(conf):
    """
        Generates object with predefined baseline model structure.

        Args:
            conf (dict): model configuration parameters

        Returns:
            model (Sequential): object with model structure
    """

    model = Sequential()

    model.add(Conv2D(16, 3, padding='same', activation=None, input_shape=(conf['height'], conf['width'], 3)))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(ReLU())
    if conf['dropout']: model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    add_conv(model, 32, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    add_conv(model, 64, conf['dropout'], conf['batch_norm'])
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation=None))
    if conf['batch_norm']: model.add(BatchNormalization())
    model.add(ReLU())
    if conf['dropout']: model.add(Dropout(0.25))

    model.add(Dense(3, activation='softmax'))

    return model
