import cv2

# The URL of your ESP32-CAM stream
stream_url = 'http://10.53.7.152:80/stream'

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open the stream. Check your IP and network connection.")
    exit()

print("Stream opened successfully. Press 'q' to quit.")

while True:
    # Read a frame from the stream
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # ---------------------------------------------
    # Insert your CV/ML processing code here
    # e.g., run inference, object detection, etc.
    # ---------------------------------------------
    
    # Display the frame
    cv2.imshow('ESP32-CAM Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()