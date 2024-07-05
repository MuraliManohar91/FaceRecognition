import cv2
import os

# Load the Haar Cascade face detector
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

Id = input('Enter User Id:')
sampleNum = 0

save_path = f"test/class{Id}"
os.makedirs(save_path, exist_ok=True)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sampleNum += 1
        # Save the captured face image in the dataset directory
        cv2.imwrite(f"{save_path}/image{sampleNum}.jpg", img[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(400)

    cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if sampleNum > 599:
        break

cam.release()
cv2.destroyAllWindows()



