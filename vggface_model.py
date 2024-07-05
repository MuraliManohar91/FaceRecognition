from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


def build_vggface_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape, name='input_1')

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1')(input_layer)
    # x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(7, 7), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
    # x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(5, 5), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
    # x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(x)
    # x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1')(x)
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2')(x)
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(x)
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(x)
    # x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Fully connected layers
    x = Flatten(name='flatten_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(1024, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    output_layer = Dense(num_classes, activation='softmax', name='dense_2')(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='VGGFace')

    return model


# Example of using the function
input_shape = (150, 150, 3)
num_classes = 5
model = build_vggface_model(input_shape, num_classes)
model.summary()