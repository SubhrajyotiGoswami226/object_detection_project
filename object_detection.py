import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from PIL import Image

def detect_objects(image_path, output_path=None, confidence_threshold=0.5):
    """
    Detect objects in an image using YOLOv8 pretrained on COCO dataset.
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the output image
        confidence_threshold (float): Minimum confidence score (0-1) for detection
    """
    # Debugging checks - STEP 2
    print(f"\n=== DEBUGGING INFORMATION ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir()}")
    print(f"Attempting to read: {os.path.abspath(image_path)}")
    print(f"File exists: {os.path.exists(image_path)}")
    
    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image not found at {image_path}")
    
    # Alternative image loading with PIL (more reliable)
    try:
        print("\nAttempting to load image with PIL...")
        pil_img = Image.open(image_path)
        image = np.array(pil_img)
        
        # Convert PIL image to OpenCV format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:  # Color (RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        print(f"Successfully loaded image with shape: {image.shape}")
    except Exception as e:
        raise ValueError(f"Error reading image: {str(e)}")

    # Load YOLOv8 model
    try:
        print("\nLoading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        print("Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {str(e)}")

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    try:
        print("\nRunning object detection...")
        results = model(image_rgb, conf=confidence_threshold)
        print(f"Found {len(results[0].boxes)} objects")
    except Exception as e:
        raise RuntimeError(f"Error during detection: {str(e)}")

    # Get class names
    class_names = model.names
    
    # Process and draw results
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = f"{class_names[class_id]}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image_rgb, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                (0, 255, 0), -1)
            
            # Put text
            cv2.putText(
                image_rgb, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Output results
    if output_path:
        try:
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"\nSuccess! Result saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving output image: {str(e)}")
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title('Object Detection Results')
        plt.show()

if __name__ == "__main__":
    try:
        # Configuration - USING YOUR SPECIFIC PATH
        input_image = r"E:\iitginternship\New folder (2)\object_detection_project\example.jpg"
        output_image = "output.jpg"  # None to display instead of saving
        
        # Run detection
        detect_objects(input_image, output_image, confidence_threshold=0.5)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)