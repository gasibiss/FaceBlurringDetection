import cv2
import numpy as np
import mediapipe as mp

# Import video
cap = cv2.VideoCapture('ex1.mp4')
mp_face = mp.solutions.face_detection

# Define GaussianBlur Function
def gaussian_kernel(size=5, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def custom_gaussian_blur(img, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma).astype(np.float32)
    blurred = np.zeros_like(img)
    for c in range(3):  # BGR
        blurred[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
    return blurred, kernel

def apply_circular_blur_feathered(original_face, blurred_face, feather):
    h, w = original_face.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.5) 

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)

    mask = mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask, mask, mask])

    blended = (blurred_face * mask_3ch + original_face * (1 - mask_3ch)).astype(np.uint8)
    return blended

def draw_kernel_on_frame(frame, kernel, pos=(10,10), scale=200):
    # kernel เป็น matrix ขนาดเล็ก เช่น 51x51
    # Normalize kernel เป็น 0-255 grayscale image
    k_min, k_max = np.min(kernel), np.max(kernel)
    kernel_img = (255 * (kernel - k_min) / (k_max - k_min)).astype(np.uint8)

    # scale kernel image ให้ใหญ่ขึ้น (เพื่อดูง่ายๆ)
    kernel_img = cv2.resize(kernel_img, (scale, scale), interpolation=cv2.INTER_NEAREST)

    # kernel_img เป็น single channel ต้องแปลงเป็น 3 channel เพื่อวางบน frame
    kernel_color = cv2.cvtColor(kernel_img, cv2.COLOR_GRAY2BGR)

    h, w = kernel_color.shape[:2]
    x, y = pos
    # วาง kernel_img บน frame
    frame[y:y+h, x:x+w] = kernel_color

with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.8) as face_detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.process(img_rgb)

        if result.detections:
            for detection in result.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = max(0, int(bboxC.xmin * iw))
                y = max(0, int(bboxC.ymin * ih))
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                x_end = min(x + w, iw)
                y_end = min(y + h, ih)

                face_roi = frame[y:y_end, x:x_end].copy()

                # Apply blur and get kernel
                blurred_face, kernel = custom_gaussian_blur(face_roi, kernel_size=51, sigma=20)

                circular_blurred_face = apply_circular_blur_feathered(face_roi, blurred_face, feather=15)
                frame[y:y_end, x:x_end] = circular_blurred_face

                # วาด kernel ที่มุมบนซ้ายของ frame (ตำแหน่งปรับได้)
                draw_kernel_on_frame(frame, kernel, pos=(10,10), scale=150)

        cv2.imshow("Custom Gaussian Face Blur with Kernel", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
