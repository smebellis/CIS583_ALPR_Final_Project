import cv2


def display_webcam():
    # Capture video from the webcam (0 for the default camera)
    cap = cv2.VideoCapture(2)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # Read the current frame from webcam
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break

        # Display the frame
        cv2.imshow("Webcam Live", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object and close display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    display_webcam()
