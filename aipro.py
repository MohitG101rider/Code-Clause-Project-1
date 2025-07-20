import cv2

# Loading files
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Load image
print("Images name - \n1. cat.png \n2. nface.png \n3. gface.jpg \n4. human.png")
print('')
a=input("Enter Image name : ")
image_path = a
image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

# Resize
image = cv2.resize(image, (600, 400))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# for human
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"Human face detected: {len(faces)}")

# for cat
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
if len(faces)==0:
    print(f"Cat face detected: {len(cats)}")
else:
    print(f"Cat face detected: 0")
    cats=cats[0:0]
    
# to detect eyes
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
print(f"Human Eyes detected: {len(eyes)}")

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for human

for (x, y, w, h) in cats:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for cat

for (x, y, w, h) in eyes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Red for eyes

# final image
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("detected_output.jpg", image)
