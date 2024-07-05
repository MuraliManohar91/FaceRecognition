# Import dataset loading script and compiled model
from load_and_preprocess import X_test, y_test
from tensorflow.keras.models import load_model

# Evaluate the model
trained_model = load_model('vggface_model.h5')
loss, accuracy = trained_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
