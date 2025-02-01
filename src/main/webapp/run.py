import cv2
from gaze_tracking import GazeTracking
import time

def main():
    # Initialize gaze tracker
    gaze_tracker = GazeTracking()
    
    # Start capturing video from webcam
    cap = cv2.VideoCapture(0)

    # Tracking variables
    total_frames = 0
    attention_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        total_frames += 1

        if not ret:
            print("Webcam not detecting face or disconnected!")
            break

        # Process gaze tracking
        frame = gaze_tracker.process_frame(frame)

        # Check if the user is looking at the screen
        if gaze_tracker.is_right() or gaze_tracker.is_left() or gaze_tracker.is_center():
            attention_frames += 1

        # Display the frame with gaze tracking
        cv2.imshow("Gaze Tracking", frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate Attention Score
    total_time = time.time() - start_time
    attention_score = (attention_frames / total_frames) * 100 if total_frames > 0 else 0

    print(f"Total Time Tracked: {total_time:.2f} seconds")
    print(f"Frames Captured: {total_frames}")
    print(f"Frames with Attention: {attention_frames}")
    print(f"Attention Score: {attention_score:.2f}%")  # User's attention percentage

    # Store in sessionStorage if using Flask (you can modify this for database logging)
    with open("attention_score.txt", "w") as file:
        file.write(f"{attention_score:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
