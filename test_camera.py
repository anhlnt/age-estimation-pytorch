import cv2
from contextlib import contextmanager
import time
from demo import draw_label

@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        print("Close camera")
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        start = time.time()
        while True:
            # get video frame
            ret, img = cap.read()
            # print(ret, img)
            if not ret:
                raise RuntimeError("Failed to capture image")
            
            # print("{:.2f}fps".format(1.0 / (time.time() - start)))
            draw_label(img, (50, 50), "{:.2f}fps".format(1.0 / (time.time() - start)))
            start = time.time()
            cv2.imshow('test', img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

yield_images()
