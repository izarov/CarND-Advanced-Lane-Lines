import sys

import numpy as np
import cv2
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from lane import Lane

def process_image(image, lane):
    lane.detect(image)
    overlay = lane.image_overlay(image.shape)
    return cv2.addWeighted(image, 1, overlay, 0.3, 0)

if __name__ == "__main__":
    l = Lane()
    video_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    processed_clip = clip1.fl_image(lambda img: process_image(img, l))

