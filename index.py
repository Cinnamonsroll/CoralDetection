import cv2
from ultralytics import YOLO


yolo_models = {
    'model1': YOLO('./algae.pt'),
    'model2': YOLO('./coral.pt')
}


videoCap = cv2.VideoCapture(0)

def get_colors(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        (base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors))) % 256
        for i in range(3)
    ]
    return tuple(color)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    
    detection_frame = frame.copy()

    for model_name, yolo in yolo_models.items():
        results = yolo.track(detection_frame, stream=True)

        for result in results:
            
            classes_names = result.names

            for box in result.boxes:
                if box.conf[0] > 0.4:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    color = get_colors(cls)

                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)

                    
                    cv2.putText(
                        frame,
                        f'{class_name} {box.conf[0]:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )

    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


videoCap.release()
cv2.destroyAllWindows()