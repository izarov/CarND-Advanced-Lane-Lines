import numpy as np
import cv2

gray_processor = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
saturation_processor = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
hue_processor = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,0]


def abs_sobel(img, img_processor=gray_processor, orient='x', sobel_kernel=7, threshold=(20, 255)):
    """
    Returns a binary mask set to 1 in locations where the absolute
    Sobel gradient in `orientation` axis is in the provided threshold range.

    Gradient is scaled to be in the [0, 255] range.
    """
    img = np.copy(img)
    img = img_processor(img)
    sobel = cv2.Sobel(img, cv2.CV_64F, 
                      1 if orient == 'x' else 0, 
                      0 if orient == 'x' else 1,
                      None, sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel, np.uint8)
    binary_output[(scaled_sobel>=threshold[0]) & (scaled_sobel<=threshold[1])] = 1

    return binary_output


def magnitude(img, img_processor=gray_processor, sobel_kernel=7, threshold=(50, 255)):
    """
    Returns a binary mask set to 1 in locations where the magnitude
    of x and y Sobel gradients as (sobel_x**2+sobel_y**2)**0.5
    is in the provided threshold range.

    Magnitude is scaled to be in the [0, 255] range.
    """
    img = np.copy(img)
    img = img_processor(img)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, None, sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, None, sobel_kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(mag, np.uint8)
    binary_output[(mag >= threshold[0]) & (mag <= threshold[1])] = 1

    return binary_output


def direction(img, img_processor=gray_processor, sobel_kernel=7, threshold=(0.7, 1.3)):
    """
    Returns a binary mask set to 1 in locations where the direction
    of x and y Sobel gradients as arctan(sobel_y/sobel_x)
    is in the provided threshold range.

    Direction is in the range [0, pi/2].
    """
    img = np.copy(img)
    img = img_processor(img)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, None, sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, None, sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dir_grad, np.uint8)
    binary_output[(dir_grad >= threshold[0]) & (dir_grad <= threshold[1])] = 1
    return binary_output


def hls(img, h_threshold=(0, 180), l_threshold=(0, 255), s_threshold=(0, 255)):
    """
    Returns a binary mask set to 1 in locations where each HLS image channel
    is in the provided threshold ranges.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    binary_output = np.zeros(img.shape[:2], dtype=np.uint8)
    binary_output[((hls[:,:,0] >= h_threshold[0]) & (hls[:,:,0] <= h_threshold[1])) &
                  ((hls[:,:,1] >= l_threshold[0]) & (hls[:,:,1] <= l_threshold[1])) &
                  ((hls[:,:,2] >= s_threshold[0]) & (hls[:,:,2] <= s_threshold[1]))] = 1
    return binary_output

def hue(img, threshold=(50, 180)):
    """
    Returns a binary mask set to 1 in locations where image saturation
    is in the provided threshold range.

    Hue is in the range [0, 180].
    """
    return hls(img, h_threshold=threshold)

def lightness(img, threshold=(100, 255)):
    """
    Returns a binary mask set to 1 in locations where image saturation
    is in the provided threshold range.

    Lightness is in the range [0, 255].
    """
    return hls(img, l_threshold=threshold)

def saturation(img, threshold=(50, 255)):
    """
    Returns a binary mask set to 1 in locations where image saturation
    is in the provided threshold range.

    Saturation is in the range [0, 255].
    """
    return hls(img, s_threshold=threshold)


yellow_threshold = ((0, 45), (85, 255), (95, 255))
yellow = lambda img: hls(img, *yellow_threshold)

white_threshold = ((0, 180), (195, 255), (0, 255))
white = lambda img: hls(img, *white_threshold)


def default(img):
    w = white(img)
    y = yellow(img)
    sobel_x_gray = abs_sobel(img)
    sobel_x = abs_sobel(img, img_processor=saturation_processor)
    grad_dir = direction(img, img_processor=saturation_processor)

    combined = np.zeros(img.shape[:2], dtype=np.uint8)

    combined[((w == 1) | (y == 1)) |
             ((sobel_x_gray == 1) | (sobel_x == 1)) &
             (grad_dir == 1)] = 1

    return combined
