from tensorflow.keras import backend
from tensorflow.keras import layers

from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization,\
    Activation, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, Lambda

from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from keras_applications.imagenet_utils import _obtain_input_shape


def siamese_mobilenet_v2(input_shape=None,
                         input_tensor=(None, None)):
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]
        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=True)

    if input_tensor is [None, None]:
        left_input = layers.Input(shape=input_shape)
        right_input = layers.Input(shape=input_shape)
    else:
        # if backend.is_keras_tensor(input_tensor[0]) and backend.is_keras_tensor(input_tensor[1]):
        if input_tensor[0] is not None and input_tensor[1] is not None:
            left_input = input_tensor[0]
            right_input = input_tensor[1]

        else:
            left_input = layers.Input(tensor=input_tensor[0], shape=input_shape)
            right_input = layers.Input(tensor=input_tensor[1], shape=input_shape)

    mobilenet = MobileNetV2(alpha=0.5, include_top=False, weights='imagenet')
    x = mobilenet.output
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.1)(x)
    out = Dense(128, activation='tanh')(x)

    shared_model = Model(inputs=mobilenet.input, outputs=out, name='mobilenetV2')

    encoded_left = shared_model(left_input)
    encoded_right = shared_model(right_input)

    Euc_layer = Lambda(lambda tensor: backend.abs(tensor[0] - tensor[1]))
    Euc_distance = Euc_layer([encoded_left, encoded_right])
    pred = Dense(16)(Euc_distance)
    pred = Dense(1, activation='sigmoid')(pred)
    model = Model(inputs=[left_input, right_input], outputs=pred)
    return shared_model, model
