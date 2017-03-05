##Advanced Lane Finding

####The goal of this project is to identify the lane boundaries and lane curvature in a video from a single front-facing camera on a car using a classic computer vision pipeline.

---

**Advanced Lane Finding Project**

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[camera]: ./images/camera_calibration.png "Camera Calibration"
[undistorted]: ./images/undistorted.png "Road Undistorted"
[binary]: ./images/binary.png "Binary Example"
[perspective]: ./images/perspective.png "Perspective Transform Example"
[histogram]: ./images/histogram.png "Histogram Search"
[search_windows]: ./images/search_windows.png "Search Windows"
[local_search]: ./images/local_search.png "Local Search"
[overlay]: ./images/overlay.png "Overlay of Fitted Lane on Image"
[video1]: ./submission.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

In this section I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The production code for this step can be found in camera.py. An interactive version of this code can be found in the CameraCalibration.ipynb notebook.

First, an array of "object points" is prepared, which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image are successfully detected.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![camera]

The camera matrix and distortion coefficients are saved in cameramatrix.p.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

The camera calibration process described above is implemented in the camera module. To undistort a raw image:

```python
import matplotlib.image as mpimg

from camera import undistort

undistort(mpimg.imread('test_images/test1.jpg'))
```

![undistorted]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate thresholded binary images (thresholds.py, lines 117-130).

The following threshold utility functions are available in the thresholds module:

* `abs_sobel` absolute Sobel gradient
* `magnitude` magnitude of x and y Sobel gradients
* `direction` direction (in radians) of x and y Sobel gradients
* `hls` convert to HLS and threshold based on hue, lightness and/or saturation

Sobel gradients are computed on a 2D image as a function of adjacent pixels. I used a Sobel kernel of size 7. The implemented Sobel functions allow for an optional pre-processor, which is a lambda function that returns a 2D version of an input 3D RGB image. Sobel gradient can differ depending on the pre-processing step to extract a 2D image from a 3D RGB image. I found that an OR combination of Sobel gradient on grayscale and saturation tends to work best.

For my image processing pipeline I use a combination of color (white and yellow) and gradient thresholds:

* absolute Sobel gradient on grayscale image (range [20, 255])
* absolute Sobel gradient on saturation channel (range [20, 255])
* color thresholds on white color ranges (hue in [0, 45], lightness in [85, 255], saturation in [95, 255])
* color threshold on yellow color ranges (hue in [0, 180], lightness in [195, 255], saturation in [0, 255])
* direction of x and y Sobel gradients (range [50, 255])

Pixels are selected according to the binary expression:

```((white OR yellow) OR (sobel_x_grayscale OR sobel_x_saturation)) AND direction```

The production threshold filter can be found in thresholds module, function `default` on line 117. An interactive version of the code, as well as examples of various combination thresholds, can be found in Thresholds.ipynb.

```python
import matplotlib.image as mpimg

from camera import undistort
import thresholds

image = undistort(mpimg.imread('test_images/test1.jpg'))
binary = thresholds.default(image)
```

Here's an example of my output for this step:

![binary]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in the perspective module (perspective.py), with an interactive version in Perspective.ipynb.

To choose the source points I started with a test image with a straight lane ahead. I chose 4 points which form a rectangle when looked at top-down and captured a large part of the image.

| Source          | Destination     |
| :-------------: | :-------------: |
| 0, 626          | 0, 626          |
| 562, 444        | 0, 0            |
| 717, 444        | 717, 0          |
| 1279, 626       | 1279, 626       |


```python
import matplotlib.image as mpimg

from camera import undistort
import perspective

image = undistort(mpimg.imread('test_images/test1.jpg'))
warped = perspective.transform(image)
back = perspective.inverse_transform(warped)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![perspective]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First, each raw image is pre-processed as explained above (lane.py, L129). Once a thresholded birds-eye view image is ready, the pipeline searches for the pixels which are part of the lane lines. The search is carried out in two different ways, depending on whether this is a single image or part of a video:

#####4a Histogram Search: Single image or detection recovery.
*This search algorithm is used when this is the first image the pipeline has seen or if fitting the lane lines failed for a number of consecutive frames.*

The histogram search algorithm first constructs a histogram of the number of non-zero pixels in the bottom half of the image. It then takes the highest peaks on the left and right half of the histogram as the left and right x coordinate starting point. The code can be found in lane.py lines 47-107.

![histogram]

The image is then split into 9 parts vertically and the algorithm proceeds to search each part in a sliding window fashion. Using the two x starting points the algorithm starts at the bottom and marks all non-zero pixels within a margin of 100 pixels to the left and right of the starting point as part of the lane lines. 

The algorithm then proceeds with the second window. If at least 50 non-zero pixels were found to be in range or "belong" to the line in the previous window, then the algorithm uses the average x coordinates of the these points as the new starting position.

![search_windows]

####4b Local Search: detection in a sequence
*This search algorithm is used after there has been a successful detection in the previous frames.*

The local search algorithm takes the fitted line estimated using histogram search and uses that as starting position for the proximity search (lane.py, line 109-123). It uses a smaller search margin (50 pixels) to mark adjacent pixels in the new frame as belonging to the left or right lane line.

![local_search]

Once either histogram or local search is run we have x and y coordinates of the pixels marked as belonging to the left and right lane line (lane.py, L143). Using a standard least squares procedure I then fit a 2nd order polynomial through these points to obtain a continuous line and a model from the cloud of pixels (yellow line in the images above).

I then use a number of configurable heuristics to check the quality of the data and fit:

* The algorithm checks if the lane width is within the expected range and if not the lane width and offset estimates (section 5) are not updated (lines 149-156).
* The curvature estimate is checked to be at least the expected minimum, using [US state specifications](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC).
* The curvature estimate is smoothed by averaging over the estimated curvature of left and right lines (lines 165-170), if the estimates are above the minimum.
* The 2nd order coefficient in the polynomial fit is checked against the previous estimate. When the difference is above a threshold (0.001) the new fit is rejected and the previous good fit is used (lines 177-189).

The pixel coordinates and polynomial fit for each frame are stored for a configurable number of iterations (lane.py 207-208).

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the real radius of the lane it is necessary to adjust the units from pixels to meters (line.py, lines 57-62). Using a polynomial fit we can then calculate the radius using the [radius to curvature equation](https://en.wikipedia.org/wiki/Radius_of_curvature) on line 63.

The offset from the center of the lane is calculated in lane.py, line 155. To calculate the width of the lane I use the average x coordinates of the marked pixels as described in the previous section. The offset is calculated as `leftx_avg + lane_width/2 - nx`, where nx is the width of the image in number of pixels.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane fit obtained in the previous steps is averaged over a configurable number of historical data points to obtain a smoother estimate of the lane position (lane.py, lines 210-225).

![overlay]

The code is organized so that it is very easy to to obtain this result from each image:

```python
import matplotlib.image as mpimg
from lane import Lane

image = mpimg.imread('test_images/test2.jpg')

l = Lane()
l.detect(image)

overlay = l.image_overlay(image.shape)
result = cv2.addWeighted(image, 1, overlay, 0.3, 0)
plt.imshow(result)
```

The code for `image_overlay` can be found in lane.py lines 227-250.

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./submission.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

An advantage of using classical computer vision methods to obtain the position of the lane is that it is easy to explain and reason about. When I contrast this with deep learning, neural nets arguably offer a black box approach to coming up with a model for purposes like these. However, engineering the features or steps of the pipeline involved quite a bit of testing and tuning parameters by hand and I feel like a pipeline with a deep learning component can be made much more robust with less effort than a classical computer vision approach alone.

The pipeline is robust to small changes in light and color shades, however not robust enough to generalize correctly on the challenge videos, where the lane has a "third" line in the middle in the first video, or has very sharp turns with limited visibility in the second one. The model is easily configurable and can benefit from further tuning of the meta-parameters.

One area that I feel can be significantly improved is the estimate obtained over the last n points. An approach using a better smoothing/filtering method, such as the Kalman filter, is likely to improve the stability of the fit. 

Another area is better outlier detection. Outliers can currently have too big of an effect on the estimates. I have left some additional checks in the code which are currently commented as they need more testing to be configured correctly. More can be done in filtering the pixels marked as belonging to each line to remove outliers. 

Overall a very interesting project to work on and I may revisit this at a later date. It would be interesting to see if convolutional neural networks can be used to help just with line pixel detection to improve detection of outliers, while keeping the rest of the pipeline the same. Another idea would be to use an ensemble of classical computer vision and CNN models.
