import sys
import cv2
from ultralytics import YOLO

HELP_TEXT = [
    "'1' - segmentation mode",
    "'2' - detection mode",
    "'3' - pose mode",
    "'e' - toggle model size",
    "      * n - less accuracy, faster",
    "      * x - more accuracy, slower",
    "'w'/'s' - increase/decrease threshold",
    "          to show boxes",
    "'q' - quit"
]

if __name__ == '__main__':
    camera = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Loading camera {camera}...")
    cap = cv2.VideoCapture(camera)

    print("Loading models...")
    models = {
        'n': {
            'Detection': YOLO('yolov8n.pt'),
            'Segmentation': YOLO('yolov8n-seg.pt'),
            'Pose': YOLO('yolov8n-pose.pt')
        },
        'x': {
            'Detection': YOLO('yolov8x.pt'),
            'Segmentation': YOLO('yolov8x-seg.pt'),
            'Pose': YOLO('yolov8x-pose.pt')
        }
    }

    threshold = 0.25
    mode = 'Segmentation'
    model_size = 'n'
    show_help = False

    print("Launching...")
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = models[model_size][mode](frame, conf=threshold)
            annotated_frame = results[0].plot()

            cv2.rectangle(annotated_frame, (0, 0), (2000, 30), (0, 0, 0), cv2.FILLED)
            cv2.putText(annotated_frame, f"Threshold: {threshold:.2}  {mode} ({model_size})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Press 'h' for help", (485, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if show_help:
                cv2.rectangle(annotated_frame, (0, 30), (325, 225), (0, 0, 0), cv2.FILLED)
                for i, line in enumerate(HELP_TEXT):
                    cv2.putText(annotated_frame, line, (10, 50 + 20 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("YOLO Live Camera Demo", annotated_frame)

            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("w"):
                threshold += 0.05
            if keypress == ord("s"):
                threshold -= 0.05
            if keypress == ord("q"):
                break
            if keypress == ord("e"):
                model_size = 'x' if model_size == 'n' else 'n'
            if keypress == ord("h"):
                show_help = not show_help
            if keypress == ord("1"):
                mode = 'Segmentation'
            if keypress == ord("2"):
                mode = 'Detection'
            if keypress == ord("3"):
                mode = 'Pose'

            if threshold > 1:
                threshold = 1.0
            elif threshold < 0:
                threshold = 0.0
        else:
            print("Failed to read frame, shutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()
