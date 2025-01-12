import cv2
import numpy as np
import os

lower_coral = np.array([0, 0, 180])
upper_coral = np.array([180, 50, 255])

def detect_coral_in_directory(directory_path: str):
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        return
    
    output_directory = os.path.join(directory_path, "output")
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            image_path = os.path.join(directory_path, filename)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to load image {filename}")
                continue

            
            image_resized = cv2.resize(image, (640, 480))
            original_image = image_resized.copy()

            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

            mask_inv = cv2.bitwise_not(mask)
            image_resized = cv2.bitwise_and(image_resized, image_resized, mask=mask_inv)

            hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
            coral_mask = cv2.inRange(hsv, lower_coral, upper_coral)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            coral_mask = cv2.morphologyEx(coral_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(coral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            height, width = coral_mask.shape
            margin = 130

            def is_near_edge(contour):
                x, y, w, h = cv2.boundingRect(contour)
                return (
                    x < margin
                    or y < margin
                    or (x + w) > (width - margin)
                    or (y + h) > (height - margin)
                )

            filtered_contours = [c for c in contours if not is_near_edge(c)]

            largest_contour = max(filtered_contours, key=cv2.contourArea, default=None)
            if largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(
                    original_image,
                    "Coral",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )
            original_image[mask_inv == 0] = [255, 255, 255]
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, original_image)
            print(f"Processed {filename} and saved result to {output_path}")
directory_path = "images"
detect_coral_in_directory(directory_path)