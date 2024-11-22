import cv2
import numpy as np
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11x.pt')
    for r in model.predict(
        source='resources/test-drone_s.mp4',
        show=True,
        conf=0.1,
        save=True,
        stream_buffer=True,
        vid_stride=4,
        classes=[0],
        stream=True
    ):
        pass
