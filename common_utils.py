from pydantic import BaseModel
import numpy as np
from typing import List, Tuple

def extract_face_coords(input_result):
    faces = []
    for result in input_result:
        classes = result.boxes.cls.cpu().tolist()
        xyxys = result.boxes.xyxy.cpu().tolist()
        for idx, xyxy in enumerate(xyxys):
            x1, y1, x2, y2 = [x for x in xyxy]
            face = (x1, y1, x2, y2)
            faces.append(face)
    return faces
def crop_from_coords(coords: List[Tuple[float, float, float, float]], 
                    image: np.ndarray) -> List[np.ndarray]:
    """
    Crop image regions based on bounding box coordinates
    
    Args:
        coords: List of tuples with (x1, y1, x2, y2) coordinates
        image: Input image as numpy array
    
    Returns:
        List of cropped image regions as numpy arrays
    """
    cropped_images = []
    
    for x1, y1, x2, y2 in coords:
        # Convert to integers and ensure proper bounds
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Clamp coordinates to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Crop the region
        cropped = image[y1:y2, x1:x2].copy()
        cropped_images.append(cropped)
    
    return cropped_images

