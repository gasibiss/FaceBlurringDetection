import cv2

# โหลด Haar Cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# โหลดภาพและย่อขนาด
image = cv2.imread('c.jpg')
im = cv2.resize(image, (600, 600))

# แปลงภาพเป็น grayscale (จำเป็นสำหรับ Haar Cascade)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# ตรวจจับใบหน้า
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# วาดสี่เหลี่ยมรอบใบหน้าที่ตรวจพบ
for (x, y, w, h) in faces:
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

# แสดงภาพ
cv2.imshow('Face Detection', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
