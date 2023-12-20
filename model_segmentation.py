from ultralytics import YOLO
import numpy as np
import cv2


def convert_to_binary(img_grayscale, thresh=100):
    thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
    return img_binary


class ModelSegmentation:
    def __init__(self):
        self.model = YOLO('Model/yolov8m-seg.pt')
        self.model.fuse()

    def predict(self, img):
        """
        return binary segmentation image (numpy array with 0, 1)
        """
        results = self.model.predict(img)

        masks = [m.data[0].numpy() for m in results[0].masks]
        segment_img = np.zeros((masks[0].shape[0], masks[0].shape[1]))

        for u in masks:
            segment_img = segment_img + u

        bin_segment_img = convert_to_binary(segment_img, 0)
        return bin_segment_img / 255.0
