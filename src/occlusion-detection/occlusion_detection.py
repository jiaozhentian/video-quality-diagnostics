import cv2
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Occlusion Detection')
    parser.add_argument('--input', '-i', type=str, default='./data/occlusion/test5.jpg', help='Input video')
    parser.add_argument('--output', '-o', type=str, default='./temp/test.jpg', help='Output video')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Threshold for occlusion detection')
    return parser.parse_args()

class OcclusionDetection(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def occlusion_detection_with_gray(self, image_src):
        # Convert to grayscale
        image_gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
        # Set the pixel value to 0 or 255 if the grayscale value is greater than 127
        _, binary_image = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # get the largest connected component
        # cv2.imshow('binary_image', binary_image)
        # cv2.waitKey(0)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
        max_area = 0
        max_area_label = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_area_label = i
        # Calculate the ratio of the largest connected component in the image
        ratio = max_area / (image_src.shape[0] * image_src.shape[1])
        if ratio > self.threshold:
            print('Ratio: ', ratio)
            
            return True
        else:
            print('Ratio: ', ratio)
            return False
    
    def occlusion_detect_with_leaf(self, image_src):
        # Convert to HSV color space
        image_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
        # Extract the H channel
        image_h = image_hsv[:, :, 0]
        low_threshold = (45, 45, 5)
        upper_threshold = (255, 255, 255)

        image_h_mask = (image_h > 35) & (image_h < 90)
        hsv_mask = cv2.inRange(image_hsv, low_threshold, upper_threshold) / 255
        tree_mask = np.asarray(image_h_mask * hsv_mask, dtype=np.uint8)
        
        # morphology operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_OPEN, kernel)
        tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imshow('tree_mask', tree_mask*255)
        cv2.waitKey(0)
        # Calculate the ratio of the tree mask component in the image
        ratio = np.sum(tree_mask) / (image_src.shape[0] * image_src.shape[1])
        if ratio > self.threshold:
            print('Ratio: ', ratio)
            return True
        else:
            print('Ratio: ', ratio)
            return False

if __name__ == '__main__':
    args = get_args()
    image_src = cv2.imread(args.input)
    occlusion_detection = OcclusionDetection(args.threshold)
    # result = occlusion_detection.occlusion_detection_with_gray(image_src)
    result = occlusion_detection.occlusion_detect_with_leaf(image_src)
    print(result)
    cv2.imwrite(args.output, image_src)
    cv2.imshow('image', image_src)
    cv2.waitKey(0)