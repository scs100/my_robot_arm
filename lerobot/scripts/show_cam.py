import cv2
import platform
from pathlib import Path

def list_available_cameras():
    """List all available camera devices on the system."""
    print("Searching for available cameras...")
    
    if platform.system() == "Linux":
        # On Linux, scan /dev/video* devices
        possible_ports = [str(port) for port in Path("/dev").glob("video*")]
        available_cameras = []
        
        for port in possible_ports:
            camera_idx = int(port.removeprefix("/dev/video"))
            cap = cv2.VideoCapture(camera_idx)
            is_open = cap.isOpened()
            cap.release()
            
            if is_open:
                print(f"Camera found at index {camera_idx} ({port})")
                available_cameras.append(camera_idx)
        
        return available_cameras
    else:
        # On other platforms, scan indices 0-10
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            is_open = cap.isOpened()
            cap.release()
            
            if is_open:
                print(f"Camera found at index {i}")
                available_cameras.append(i)
        
        return available_cameras

# Find available cameras
available_cameras = list_available_cameras()

if not available_cameras:
    print("No cameras found. Please check your connections.")
    exit(1)

# Try to use the first available camera
camera_index = available_cameras[0]
print(f"Attempting to open camera at index {camera_index}")

# On Linux, use V4L2 backend explicitly
if platform.system() == "Linux":
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Failed to open camera at index {camera_index}")
    exit(1)

print(f"Successfully opened camera at index {camera_index}")
print("Press 'q' to quit, 'n' to try next camera")

current_camera_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break
    
    cv2.imshow(f'Camera Feed (Index: {available_cameras[current_camera_idx]})', frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    elif key == ord('n') and len(available_cameras) > 1:
        # Switch to next camera
        cap.release()
        current_camera_idx = (current_camera_idx + 1) % len(available_cameras)
        camera_index = available_cameras[current_camera_idx]
        print(f"Switching to camera index {camera_index}")
        
        if platform.system() == "Linux":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(camera_index)

cap.release()
cv2.destroyAllWindows()
