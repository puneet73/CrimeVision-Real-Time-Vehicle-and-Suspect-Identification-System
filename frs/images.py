import cv2
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the video file
cap = cv2.VideoCapture('E:\\1 face source\\video.mp4')

# Get the video frame rate
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (640, 360))  # set the video dimensions here

# Specify the output directory for the extracted faces
output_dir = 'output_faces'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize a counter for the extracted faces
count = 0

# Loop through each frame of the video
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()

    # If there are no more frames, exit the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face and extract it
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Save the face to the output directory
        cv2.imwrite(os.path.join(output_dir, f'face_{count}.jpg'), face)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Increment the counter
        count += 1

    # Write the frame with the detected faces to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('frame', frame)

    # Wait for the 'q' key to be pressed to stop the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and the output video writer, and close the window
cap.release()
out.release()
cv2.destroyAllWindows()
