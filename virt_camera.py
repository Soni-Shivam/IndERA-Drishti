import cv2
import pyvirtualcam

# The correct endpoint for the raw video stream
stream_url = 'http://10.53.7.152:81/stream'

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print(f"Error: Could not open the ESP32 stream at {stream_url}")
    print("Check if the IP is correct and the ESP32 is powered on.")
    exit()

# Get the resolution of the ESP32 stream
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Target FPS
target_fps = 20 

print(f"Connecting to stream...")
print(f"Starting Virtual Camera at {width}x{height}...")

# Start the virtual camera
with pyvirtualcam.Camera(width=width, height=height, fps=target_fps) as cam:
    print(f'Virtual camera created: {cam.device}. You can now select this in Zoom, OBS, or Chrome.')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from ESP32. Retrying...")
            continue

        # OpenCV captures in BGR format, but virtual cameras expect RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Send the frame to the virtual camera
        cam.send(frame_rgb)
        
        # Sleep to maintain the target FPS and not overwhelm the system
        cam.sleep_until_next_frame()

cap.release()