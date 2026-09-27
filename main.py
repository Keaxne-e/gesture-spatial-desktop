import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read frame.")
            break

        cv2.imshow("Gesture Spatial Desktop", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()