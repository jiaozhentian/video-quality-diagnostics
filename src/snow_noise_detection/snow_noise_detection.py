import cv2
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Snow Noise Detection')
    parser.add_argument('--input', '-i', type=str, default='./data/snow_noise/test6.jpg', help='Input video')
    parser.add_argument('--output', '-o', type=str, default='./temp/test.jpg', help='Output video')
    parser.add_argument('--center-rate', '-c', type=float, default=0.4, help='Threshold for occlusion detection')
    parser.add_argument('--threshold', '-t', type=float, default=60, help='Threshold for occlusion detection')
    return parser.parse_args()

class OcclusionDetection(object):
    def __init__(self, center_rate=0.1, threshold=60):
        self.threshold = threshold
        self.center_rate = center_rate

    def detect(self, image_src):
        image_gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        (x_center, y_center) = (image_gray.shape[1] // 2, image_gray.shape[0] // 2)
        cv2.imshow('image_gray', image_gray)
        cv2.waitKey(0)
        
        fft = np.fft.fft2(image_gray)
        fft_shift = np.fft.fftshift(fft)

        mask = np.zeros_like(image_gray, dtype=np.float32)
        mask[y_center - int(self.center_rate * image_gray.shape[1]):
             y_center + int(self.center_rate * image_gray.shape[1]),
                x_center - int(self.center_rate * image_gray.shape[0]):
                x_center + int(self.center_rate * image_gray.shape[0])] = 1
        fft_shift = fft_shift * mask

        ifft_shift = np.fft.ifftshift(fft_shift)
        recon = np.fft.ifft2(ifft_shift)

        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        if mean > self.threshold:
            print('Noise: {}'.format(mean))
            return True
        else:
            print('Not Noise: {}'.format(mean))
            return False

if __name__ == '__main__':
    args = get_args()
    # Read the image
    image_src = cv2.imread(args.input)
    occlusion_detection = OcclusionDetection(center_rate=args.center_rate, threshold=args.threshold)
    occlusion_detection.detect(image_src)
    cv2.imshow('image', image_src)
    cv2.waitKey(0)
