# Import dataset loading and model compilation scripts
from load_and_preprocess import X_train, y_train, X_test, y_test
from compile_model import model

# Training parameters
batch_size = 4
epochs = 30

# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Save the model
model.save('vggface_model.h5')
