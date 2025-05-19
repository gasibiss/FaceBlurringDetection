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
    # print(kernel)
    return kernel

def custom_gaussian_blur(img, kernel_size=5, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma).astype(np.float32)
    blurred = np.zeros_like(img)
    for c in range(3):  # BGR
        blurred[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
    print(blurred)
    return blurred

def apply_circular_blur_feathered(original_face, blurred_face, feather):
    # print("feather value :",feather)
    h, w = original_face.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.5) 

    # Create White Mask to be circle in black area
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Making Feather edge blur with GaussianBlur
    mask = cv2.GaussianBlur(mask, (feather*2+1, feather*2+1), 0)

    # Convert Maske to be 3 channels
    mask = mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask, mask, mask])

    # Blend
    blended = (blurred_face * mask_3ch + original_face * (1 - mask_3ch)).astype(np.uint8)
    return blended




# Start detecting
with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.8) as face_detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        # BGR → RGB (For MediaPipe)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Contain object detection
        # Entry face detector
        result = face_detector.process(img_rgb)

        if result.detections:
            for detection in result.detections:

                # BoxxC is Face Detected bounding box in "relative" format
                # BoxxC Pull position of face
                bboxC = detection.location_data.relative_bounding_box

                # size of Image
                ih, iw, _ = frame.shape

                #  ( xmin, ymin, width, height ) is ratio
                #  0 is prevent negative value
                #  Multiply to be real Pixel
                x = max(0, int(bboxC.xmin * iw))
                y = max(0, int(bboxC.ymin * ih))
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                x_end = min(x + w, iw)
                y_end = min(y + h, ih)

                # Step 1: Extract face region (ROI)
                # ROI --> Region of Interest
                # .copy() is for not adjust directly frame
                face_roi = frame[y:y_end, x:x_end].copy()

                # Step 2: Apply custom Gaussian blur
                blurred_face = custom_gaussian_blur(face_roi, kernel_size=51, sigma=10)

                # Step 3: Paste blurred face back to frame
                # frame[y:y_end, x:x_end] = blurred_face
                circular_blurred_face = apply_circular_blur_feathered(face_roi, blurred_face,feather=15)
                frame[y:y_end, x:x_end] = circular_blurred_face


        cv2.imshow("Custom Gaussian Face Blur", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
