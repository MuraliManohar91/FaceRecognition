import tensorflow as tf
import numpy as np
import cv2
import pickle

# Placeholder for loading the fine-tuned VGGFace model
recognition_model = tf.keras.models.load_model("vggface_model.h5")

# Load the label map
with open("label_map.pickle", "rb") as f:
    label_map = pickle.load(f)

# Invert the label map to get a mapping from indices to names
labels = {v: k for k, v in label_map.items()}

def recognize_faces(faces, gray_frame):
    recognized_faces = []
    accuracies = []

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]

        if face is None or not isinstance(face, np.ndarray):
            print("Invalid face image.")
            continue

        try:
            face = cv2.resize(face, (150, 150))
        except cv2.error as e:
            print(f"Error resizing face: {e}")
            continue
        #face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.astype("float32") / 255.0
        #face = np.expand_dims(face, axis=-1)  # Adding channel dimension for grayscale
        face = np.expand_dims(face, axis=0)

        preds = recognition_model.predict(face)[0]
        j = np.argmax(preds)
        label = labels[j]
        accuracy = preds[j] * 100

        recognized_faces.append(label)
        accuracies.append(accuracy)

    return recognized_faces, accuracies

# Testing face recognition
if __name__ == "__main__":

    # Test with a real image from the dataset to verify the label map
    test_image_path = 'test/class1/image6.jpg'  # Replace with actual path
    test_image = cv2.imread(test_image_path)
    faces = [test_image]
    recognized_faces, accuracies = recognize_faces(faces)
    for label, accuracy in zip(recognized_faces, accuracies):
        print(f"Recognized: {label} with accuracy: {accuracy:.2f}%")
