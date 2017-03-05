import logging
import numpy as np

class Line:
    """
    The Line class is a container for the characteristics of each line detection
    """
    MAX_HISTORY = 100 # max nr of frames to keep information for
    MIN_RADIUS = 200 # min radius in meters to be expected for a US road

    def __init__(self):
        # was the line detected in the last n iterations (T/F)
        self.detections = []
        # array of fit coefficients for the last n iterations
        self.fits = []
        #x values for detected line pixels in the last n iterations
        self.allx = []
        #y values for detected line pixels in the last n iterations
        self.ally = []
        #radius of curvature of the line in some units
        self._radius_of_curvature = None

    def detected(self, n=5):
        """
        Returns True if there was a successful detection (True)
        in the last `n` iterations.
        """
        return np.sum(np.bool8(self.detections[-n:])) > 0

    @property
    def radius_of_curvature(self):
        return self._radius_of_curvature

    @radius_of_curvature.setter
    def radius_of_curvature(self, value):
        self._radius_of_curvature = value

    def _pop_from_history(self, i=0):
        ls = [self.allx, self.ally]
        for l in ls:
            l.pop(i)

    def add_detection(self, xs, ys, fit, radius, offset, good_fit=True):
        """
        Add detection characteristics.
        """
        if len(self.allx) > self.MAX_HISTORY:
            self._pop_from_history()

        self.fits.append(fit)
        self.allx.append(xs)
        self.ally.append(ys)
        self.radius_of_curvature = radius
        self.detections.append(good_fit)


def curve_radius(y, xs, ys, lane_height_px=300, lane_width_px=700):
    ym_per_px = 30/lane_height_px # meters per pixel in y dimension, visible lane ~ 30m
    xm_per_px = 3.7/lane_width_px # meters per pixel in x dimension

    # Fit new polynomials to y, x in world space
    fit = np.polyfit(ys*ym_per_px, xs*xm_per_px, 2)
    radius = ((1 + (2*fit[0]*y*ym_per_px + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return radius
