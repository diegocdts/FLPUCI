from helpers.types_helper import FCAEProperties

AE_PROPERTIES = FCAEProperties(input_shape=(1, 12, 1), encode_layers=[32], encode_activation='relu',
                               decode_activation='linear', kernel_size=(3, 3), strides=2, padding='same',
                               latent_space=8, learning_rate=0.0005)
