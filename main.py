import cv2
import mediapipe as mp
import time
import math 
import webbrowser

MODEL_PATH = "models/hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),             # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             # index
    (5, 9), (9, 10), (10, 11), (11, 12),        # middle
    (9, 13), (13, 14), (14, 15), (15, 16),      # ring
    (13, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (0, 17)                                     # wrist
]

PINCH_START_THRESHOLD = 0.25
PINCH_RELEASE_THRESHOLD = 0.40


def main():
    
    pinch_frames = 0
    pinch_active = False

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH), #load the model from the specified path
        running_mode=RunningMode.VIDEO, #treat the input as a video stream of frames
        num_hands=2
    )

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image( #wrap the frame in a mediapipe Image object
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video( #runs the hand landmark detection model on the frame
                mp_image,
                timestamp_ms
            )

            height, width, _ = frame.shape

            if result.hand_landmarks:                   
                for hand_landmarks in result.hand_landmarks:
                    points = []
                    points_3d = []
                
                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        z = landmark.z

                        points.append((x, y))           #store the pixel coordinates of the landmarks in a list to referebce later
                        points_3d.append((landmark.x, landmark.y, landmark.z))

                    thumb_tip_3d = points_3d[4]
                    index_tip_3d = points_3d[8]

                    pinch_distance = math.dist(thumb_tip_3d, index_tip_3d)

                    # Use wrist to middle finger base as a rough hand size reference
                    hand_size = math.dist(points_3d[0], points_3d[9])
                    pinch_ratio = pinch_distance / hand_size
                    cv2.putText(frame, f"Pinch Ratio: {pinch_ratio:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    

                    if not pinch_active:
                        if pinch_ratio < PINCH_START_THRESHOLD:
                            pinch_frames += 1
                        else:
                            pinch_frames = 0

                        if pinch_frames >= 5:           #temporal filtering to avoid false positives, only trigger if pinch is detected for 5 consecutive frames
                            pinch_active = True
                            pinch_frames = 0
                            webbrowser.open("https://www.netflix.com")  

                    
                    else:
                        if pinch_ratio > PINCH_RELEASE_THRESHOLD:
                            pinch_active = False

                    if pinch_active:
                        cv2.putText(frame, "Pinch Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    for start, end in HAND_CONNECTIONS:
                        cv2.line(                       #draw lines between the landmarks for skeleton
                            frame,
                            points[start],      
                            points[end],
                            (255, 255, 255),
                            2
                        )   

                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

            cv2.imshow("Gesture Spatial Desktop", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()