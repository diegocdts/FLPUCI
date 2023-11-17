from helpers.types_helper import FCAEProperties

AE_PROPERTIES = FCAEProperties(input_shape=(4, 12, 1), encode_layers=[128, 64], encode_activation='relu',
                               decode_activation='linear', kernel_size=(5, 5), encode_strides=[2, 2], padding='same',
                               latent_space=12, learning_rate=0.0001)
