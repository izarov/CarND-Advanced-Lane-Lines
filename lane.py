import logging
import numpy as np
import cv2

from camera import undistort
import thresholds
import perspective
from line import Line, curve_radius

class Lane:
    """
    The Lane class is responsible for detecting the lane and keeping history in (a series of) images.
    """

    # Number of sliding search windows
    SEARCH_WINDOWS = 9
    # Width of window search margin (+/-)
    MARGIN = 100
    # Minimum number of pixels found to recenter window
    MIN_PIXELS = 50
    # Lane highlight color
    HIGHLIGHT_COLOR = (0, 255, 0) # green

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = 0. # car offset in pixels from middle of the lane
        self.lane_width = 0. # estimated lane width in pixels
        self.curvature = 0. # estimated curvature in meters
        self.left = Line()
        self.right = Line()
        return self

    @staticmethod
    def process_image(image):
        """
        Image pre-processing pipeline.
        """
        image = np.copy(image)
        image = undistort(image)
        image = thresholds.default(image)
        image = perspective.transform(image)
        return image


    def histogram_search(self, image):
        """
        Takes a pre-processed image and steps through SEARCH_WINDOWS windows one by one.

        Initial position is estimated using a histogram of the bottom half of the image.

        This is a modified version of the code in SDCND's Advanced Lane Finding module, 33. Finding the Lines
        """
        margin = self.MARGIN
        nwindows = self.SEARCH_WINDOWS
        minpix = self.MIN_PIXELS

        ny, nx = image.shape
        window_height = ny//nwindows
        # compute a histogram of the bottom half of the image
        histogram = np.sum(image[ny//2:,:], axis=0)
        midpoint = len(histogram)//2
        # split histogram array in the middle to find left and right pixel max
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Current lane line positions to be updated for each window
        leftx_current, rightx_current = leftx_base, rightx_base

        nonzeroy, nonzerox = image.nonzero()
        left_rectangles = []
        right_rectangles = []
        left_lane_inds = []
        right_lane_inds = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = ny - (window+1)*window_height
            win_y_high = ny - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            left_rectangles.append([(win_xleft_low,win_y_low),(win_xleft_high,win_y_high)])
            right_rectangles.append([(win_xright_low,win_y_low),(win_xright_high,win_y_high)])

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if we find > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return {'indices': (left_lane_inds, right_lane_inds),
                'search_rectangles': (left_rectangles, right_rectangles),
                'lane_base': (leftx_base, rightx_base)}

    def local_search(self, image):
        """
        Search only in areas adjacent to the previous fit.
        """
        margin = self.MARGIN//2
        nonzeroy, nonzerox = image.nonzero()
        left_fit = self.left.fits[-1]
        right_fit = self.right.fits[-1]

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        return left_lane_inds, right_lane_inds

    def detect(self, orig_image):
        """
        Detect lane in a raw image.
        """
        image = self.process_image(orig_image)

        nonzeroy, nonzerox = image.nonzero()

        if not self.left.detected() and not self.right.detected():
            # no detections, start from scratch
            res = self.histogram_search(image)
            left_inds, right_inds = res['indices']
            left_x, right_x = res['lane_base']
            self.lane_width_px = right_x - left_x
        else:
            left_inds, right_inds = self.local_search(image)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_inds]
        lefty = nonzeroy[left_inds]
        rightx = nonzerox[right_inds]
        righty = nonzeroy[right_inds]

        leftx_avg, rightx_avg = np.mean(leftx), np.mean(rightx)
        est_lane_width = (rightx_avg - leftx_avg)
        if (est_lane_width < 0.7*self.lane_width_px or
            est_lane_width > 1.3*self.lane_width_px):
            # logging.warning('lane width estimate outside of range: {}'.format(est_lane_width))
            pass
        else:
            self.offset = leftx_avg + est_lane_width/2 - image.shape[1]/2
            self.lane_width = est_lane_width

        left_radius = curve_radius(image.shape[0]-1, leftx, lefty)
        right_radius = curve_radius(image.shape[0]-1, rightx, righty)

        if abs(left_radius-right_radius)/(left_radius+right_radius) > 0.5:
            # logging.warning('too big of a difference in radius of the left and right lines of lane: {} and {}'.format(left_radius, right_radius))
            pass
        else:
            if left_radius > Line.MIN_RADIUS and right_radius > Line.MIN_RADIUS:
                self.curvature = (left_radius + right_radius)/2
            elif left_radius > Line.MIN_RADIUS:
                self.curvature = left_radius
            else:
                self.curvature = right_radius

        diff = lambda x1, x2: abs(x1-x2)/((x1 + x2)/2)

        left_fit_ok, right_fit_ok = [len(x) > self.MIN_PIXELS for x in [leftx, rightx]]

        left_fit = np.polyfit(lefty, leftx, 2)
        if (left_radius < Line.MIN_RADIUS or
            (len(self.left.fits) > 0 and abs(self.left.fits[-1][0]-left_fit[0]) > 0.001)):
            # discard new fit
            left_fit = self.left.fits[-1]
            left_fit_ok = False
        # left_r_squared = r_squared(left_fit, lefty, leftx)

        right_fit = np.polyfit(righty, rightx, 2)
        if (right_radius < Line.MIN_RADIUS or
            (len(self.right.fits) > 0 and abs(self.right.fits[-1][0]-right_fit[0]) > 0.001)):
            # discard new fit
            right_fit = self.right.fits[-1]
            right_fit_ok = False
        # right_r_squared = r_squared(right_fit, righty, rightx)

        # if (len(self.left.fits) > 0 and
            # (diff(left_fit[0], np.mean(self.left.fits[-3:][0])) > 0.9 or
             # diff(left_fit[1], np.mean(self.left.fits[-3:][1])) > 0.5)):
            # logging.warning('big difference in new left fit: {} vs old {}'.format(left_fit, self.left.fits[-1]))
            # left_fit_ok = False

        # if (len(self.right.fits) > 0 and
            # (diff(right_fit[0], np.mean(self.right.fits[-3:][0])) > 0.9 or
             # diff(right_fit[1], np.mean(self.right.fits[-3:][1])) > 0.5)):
            # right_fit_ok = False

        # if (diff(left_fit[0], right_fit[0]) > 0.9 or
        #     diff(left_fit[1], right_fit[1]) > 0.5):
        #     # logging.warning('big difference in left_fit vs right_fit: {} vs {}'.format(left_fit, right_fit))

        self.left.add_detection(leftx, lefty, left_fit, left_radius, left_fit_ok)
        self.right.add_detection(rightx, righty, right_fit, right_radius, right_fit_ok)

    def estimate(self, ys, n=5):
        """
        Using up to `n` previous fits returns an estimate of left and right x positions given `ys`.
        """
        leftx = np.zeros_like(ys)
        rightx = np.zeros_like(ys)

        for f in self.left.fits[-n:]:
            leftx += f[0]*ys**2 + f[1]*ys + f[2]

        for f in self.right.fits[-n:]:
            rightx += f[0]*ys**2 + f[1]*ys + f[2]

        n = len(self.left.fits[-n:])

        return leftx/n, rightx/n

    def image_overlay(self, image_shape):
        ret = np.zeros(image_shape[:2], np.uint8)
        ret = np.dstack((ret, )*3)

        ploty = np.linspace(0, image_shape[0]-1, image_shape[0])
        left_fitx, right_fitx = self.estimate(ploty)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(ret, np.int_([pts]), self.HIGHLIGHT_COLOR)

        # Warp the blank back to original image space using inverse perspective matrix
        ret = perspective.inverse_transform(ret)

        # convert offset to meters
        offset = self.offset*3.7/700
        ret = cv2.putText(ret, "Offset: {:.2f}m, curvature: {:.0f}m".format(offset, self.curvature),
                (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        return ret

    def warped_image(self, image):
        margin = self.MARGIN

        image = self.process_image(image)

        # Create an image to draw on and an image to show the selection window
        ret = np.dstack((image, )*3)*255
        window_img = np.zeros_like(ret)
        # Color in left and right line pixels
        ret[self.left.ally[-1], self.left.allx[-1]] = [255, 0, 0]
        ret[self.right.ally[-1], self.right.allx[-1]] = [0, 0, 255]

        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        left_fitx, right_fitx = self.estimate(ploty)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose([left_fitx-margin, ploty])])
        left_line_window2 = np.array([np.flipud(np.transpose([left_fitx+margin, ploty]))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose([right_fitx-margin, ploty])])
        right_line_window2 = np.array([np.flipud(np.transpose([right_fitx+margin, ploty]))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), self.HIGHLIGHT_COLOR)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), self.HIGHLIGHT_COLOR)

        ret = cv2.addWeighted(ret, 1, window_img, 0.3, 0)

        return ret, left_fitx, right_fitx, ploty


def r_squared(coeffs, xs, ys):
    pol = np.poly1d(coeffs)
    yhat = pol(xs)
    ybar = np.mean(ys)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((ys-ybar)**2)
    return ssreg/sstot
