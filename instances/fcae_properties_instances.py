from helpers.types_helper import FCAEProperties

AE_PROPERTIES = FCAEProperties(input_shape=(4, 56, 1),
                               encode_layers=[16, 8],
                               encode_activation='relu',
                               decode_activation='relu',
                               kernel_size=(5, 5),
                               encode_strides=[2, 2],
                               padding='same',
                               latent_space=10,
                               learning_rate=0.001)
