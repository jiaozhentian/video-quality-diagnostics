import cv2
import enum
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Luminance Anomaly Detection')
    parser.add_argument('--input', '-i', type=str, default='./data/luminace/test2.jpg', help='Input video')
    parser.add_argument('--output', '-o', type=str, default='./temp/test.jpg', help='Output video')
    parser.add_argument('--low_threshold', '-l', type=float, default=0.3, help='Threshold for anomaly detection')
    parser.add_argument('--high_threshold', '-t', type=float, default=0.7, help='Threshold for anomaly detection')
    return parser.parse_args()

class Luminace(enum.Enum):
    Normal = 0
    Low = 1
    High = 2

class LuminaceAnomalyDetection(object):
    def __init__(self, low_threshold=0.2, high_threshold=0.8):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
       
    def detect(self, image_src):
        # Convert to HSV color space
        image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
        # Extract the V channel
        image_v = image_hsv[:, :, 2]
        # Calculate the average luminance
        average_luminance = np.mean(image_v) / 255.0
        # judge the luminance ratio
        if average_luminance < self.low_threshold:
            return Luminace.Low
        elif average_luminance > self.high_threshold:
            return Luminace.High
        else:
            return Luminace.Normal

if __name__ == '__main__':
    args = get_args()
    # Read the image
    image_src = cv2.imread(args.input)
    # Create an instance of the luminance anomaly detection class
    detector = LuminaceAnomalyDetection(args.low_threshold, args.high_threshold)
    # Detect the luminance anomaly
    luminance = detector.detect(image_src)
    # Display the result
    print('Luminance: {}'.format(luminance.name))
    # Save the result
    cv2.imwrite(args.output, image_src)
    cv2.imshow('image', image_src)
    cv2.waitKey(0)
