import cv2
from databaseInsertion import insert_to_db
from reportGeneration import generate_report
from faceRecognition import recognize_faces
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        recognized_faces, accuracies = recognize_faces(faces, frame)

        for ((x, y, w, h), label, accuracy) in zip(faces, recognized_faces, accuracies):
            if accuracy > 80.0:
                insert_to_db(label, accuracy)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            t = y - 10 if y - 10 > 10 else y + 10
            text = f"{label}: {accuracy:.2f}%"
            cv2.putText(frame, text, (x, t), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Face Detection and Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate the report
    generate_report()

if __name__ == "__main__":
    main()
