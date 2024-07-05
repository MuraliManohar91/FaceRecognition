from tensorflow.keras.optimizers import Adam

# Import the model definition
from vggface_model import build_vggface_model

# Define the model
input_shape = (150, 150, 3)
num_classes = 5
model = build_vggface_model(input_shape, num_classes)


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
