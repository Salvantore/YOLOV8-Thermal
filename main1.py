from ultralytics import YOLO
import cv2

# Load a YOLO model
model = YOLO('yolov8n.pt')

# Function to draw annotations (bounding boxes, class labels, and confidence scores)
def draw_annotations(object_detection, image):
    num_people = 0
    for box in object_detection[0].boxes.data:
        class_idx = int(box[5])
        confidence = box[4].item()
        class_name = model.names[class_idx]
        if class_name == 'person':
            num_people += 1
            x1, y1, x2, y2 = map(int, box[:4])
            label = f'{class_name} {confidence:.2f}'
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Green bounding box

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw red rectangle background for text
            cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 0, 255), cv2.FILLED)

            # Put white text on the red background
            cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text
    return image, num_people

# Function to process image
def process_image(input_image_path, output_image_path):
    frame = cv2.imread(input_image_path)

    # Perform object detection
    detections = model(frame, conf=0.25)

    # Draw annotations and count number of people
    frame_with_annotations, num_people = draw_annotations(detections, frame)

    # Get image dimensions
    height, width, _ = frame_with_annotations.shape

    # Display total number of people at bottom left corner
    text = f'People: {num_people}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = 10
    text_y = height - 10  # Move up from the bottom by 10 pixels
    cv2.putText(frame_with_annotations, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save annotated image
    cv2.imwrite(output_image_path, frame_with_annotations)

    # Display annotated image
    cv2.imshow('Annotated Image', frame_with_annotations)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process image
process_image("D:/DO AN TN/DATN1/.vscode/anh sang thuong.jpg", "D:/DO AN TN/DATN1/.vscode/coas2.jpg")
