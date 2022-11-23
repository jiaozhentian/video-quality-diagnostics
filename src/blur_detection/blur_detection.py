import cv2
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Occlusion Detection')
    parser.add_argument('--input', '-i', type=str,
                        default='./data/blur/test6.jpg',
                        help='Input video')
    parser.add_argument('--output', '-o', type=str,
                        default='./temp/test.jpg',
                        help='Output video')
    parser.add_argument('--threshold', '-t', type=float,
                        default=10,
                        help='Threshold for occlusion detection')
    return parser.parse_args()

class BlurDetection(object):
    def __init__(self, threshold=10, size=60):
        self.threshold = threshold
        self.size = size

    def detect(self, image_src):
        # Convert to grayscale
        image_gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        (x_center, y_center) = (image_src.shape[1] // 2, image_src.shape[0] // 2)
        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(image_gray)
        fftShift = np.fft.fftshift(fft)

        # zero-out the center of the FFT shift (i.e., remove low
        # frequencies), apply the inverse shift such that the DC
        # component once again becomes the top-left, and then apply
        # the inverse FFT
        fftShift[y_center - self.size:y_center + self.size,
                 x_center - self.size:x_center + self.size] = 0
        
        ifftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(ifftShift)

        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        if mean < self.threshold:
            print('Blurry: {}'.format(mean))
            return True
        else:
            print('Not Blurry: {}'.format(mean))
            return False
        
if __name__ == '__main__':
    args = get_args()
    # Read the image
    image_src = cv2.imread(args.input)
    # Create an instance of the blur detection class
    detector = BlurDetection(args.threshold)
    # Detect the blur
    blur = detector.detect(image_src)
    # Display the result
    print('Blur: {}'.format(blur))
    # Save the result
    cv2.imwrite(args.output, image_src)
    cv2.imshow('image', image_src)
    cv2.waitKey(0)