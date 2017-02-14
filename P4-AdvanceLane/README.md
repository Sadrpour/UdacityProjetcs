##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./projectImages/camera.jpg "Camera Caliberation"
[image2]: ./projectImages/undistorted.jpg "Undistorted Chess"
[image3]: ./projectImages/binary.jpg "Binary Example 1"
[image4]: ./projectImages/binary2.jpg "Binary Example 2"
[image5]: ./projectImages/perspective.jpg "Perspective Example 1"
[image6]: ./projectImages/perspective2.jpg "Perspective Example 2"
[image7]: ./projectImages/hist.jpg "Histogram Example 1"
[image8]: ./projectImages/hist2.jpg "Histogram Example 2"
[image9]: ./projectImages/boundingWindows.jpg "Bounding Windows for Lane Fitting 1"
[image10=250x]: ./projectImages/boundingWindows2.jpg "Bounding Windows for Lane Fitting 2"
[image11]: ./projectImages/final1.jpg "Final Output Example 1"
[image12]: ./projectImages/final2.jpg "Final Output Example 2"
[video1]: ./project_video.mp4 "Video"

---
###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

<img src="./projectImages/camera.jpg" alt="alt text" width=400 height=300>

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

<img src="./projectImages/undistorted.jpg" alt="alt text" width=400 height=300>

The result for this section can be found in Section 1 of my code under "Camera Caliberation" and "Correcting Distortion"


####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I spent hours playing around with the src and dst points to get a reasonable mapping for the lanes. in the end, the 4 points provided in the rubric beat my own src and dst points. One of the key factors was that the src points extended across the y-axis of the image. the offset parameter which is shown in the code snippet below controls how much of the original image is included in the perspective transform. The lower the number the more streched out the final image. This means less of the original image is going to be included in the final image. This was a key parameter to exclude excessive features from going into the model. For instnace if offset was larger, we would be including cars and lanes in which the car is not driving in. 

The result for this section can be found in Section 2 of my code under "Perspective Transformation".
```
offset = 290
# in the cv2 transformation first dimension is x second is y 
# in image.shape first dimension is y then x
img_size = (img.shape[1],img.shape[0])
src = np.float32([[585,460],
                  [203,720],
                  [1127,720],
                  [695,460]])
dst = np.float32([[offset,0],
                  [offset,img_size[1]],
                  [img_size[0]-offset,img_size[1]],
                  [img_size[0]-offset,0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 290, 0        | 
| 203, 720      | 290, 720      |
| 1127, 720     | 910, 720      |
| 695, 460      | 910, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. See images below for some examples:

<img src="./projectImages/perspective.jpg" alt="alt text" width=400 height=300>
<img src="./projectImages/perspective2.jpg" alt="alt text" width=400 height=300>


####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The course provides us a vareity of tools for detecting lane lines in the images as follows: (1) gradient in x and y directions, (2) magnitute of gradient, (3) angle of gradient, (4) HSV and HLS color spaces.

I spent a lot of time playing around with various combinations of them, but my tunning did not beat the tunning i saw during the office hours. So i ended up using that. The combination of (1) large x,y gradient OR (2) large saturation (HLS) and value (HSV) channels in the HLS and HSV transformation turned out to be the most useful to cleanly detect the lane lines and colors. I am still not sure why S and V channels were the most useful channels and not for example the hue. Below are two examples that demonstrates the filters.

<img src="./projectImages/binary.jpg" alt="alt text" width=400 height=300>
<img src="./projectImages/binary2.jpg" alt="alt text" width=400 height=300>

The result for this section can be found in Section 3 of my code under "Binary Image Generation"

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For detecting lines i used the sliding window methodology introduced in the course lectures. A given image was sliced into 9 segments. A rectangular area was defined along each segment, for the left and right lanes. All the active pixels (non-zero) were identified inside each rectangle. A second order polynomial was fit into the point clouds. Finally a smoothing parameter was added that allowed me to draw the polynomial on the road using the average of n past polynomial fits rather than just the current one. The smoother is implemented in Section 4 of my code using the function "outlier_removal". This smoothed the lines and results looked more stable. 



Please see images below for some samples from this process:

image 1
(a) Histogram of points (b) Fitted polynomial, sliding windows bounds, and point clouds
<img src="./projectImages/hist1.jpg" alt="alt text" width=600 height=300>
<img src="./projectImages/poly1.jpg" alt="alt text" width=600 height=300>

image 2 
(a) Histogram of points (b) Fitted polynomial, sliding windows bounds, and point clouds
<img src="./projectImages/hist2.jpg" alt="alt text" width=600 height=300>
<img src="./projectImages/poly2.jpg" alt="alt text" width=600 height=300>

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The polynomial fits were implemented on the image scale. We then use a conversion, to map points along each lane in the image into real world scale (unit meter). a polynomial was then fit to the real-wrold points, and the closest point to the vehicle was chosen to calculate the curvature. the curvature was calculated according to the formula introduced in [Perspective Transformation](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) as suggested in the course lectures. The curvature was then added to the final image. Based on the results and video implementation
I think the curvature values are within acceptable range. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

All of the steps above was combined together into a single pipeline (please find function "lineFinder" in Section 4 of my code). Before feeding the images to the pipeline they were undistored and warped accordingly.

Images below show that the algorithm is able to identify the region of inetersted between the lane lines accurately. 

<img src="./projectImages/final1.jpg" alt="alt text" width=400 height=300>
<img src="./projectImages/final2.jpg" alt="alt text" width=400 height=300>

---

###Pipeline (video)

####1. Here is a link to my video

Here's a [link to my video result](./submission_video.mp4)

---

###Discussion

The pipeline initially failed when another vehile started to approach from the right. I had to tune the perspective transformation to make sure it does not cover too much of the original image. We are mainly concerned with the area in front of the vehicle.  I spent alot of time on perspective transformation and gradients. I wish there was a better and faster way of doing this. This pipeline does not do well on more challenging videos. I also applied an outlier removal strategy to remove polynomial fits that were very different from their previous fits. This did not improve the result because polynomials have many degrees of freedom and i did not find any threshold that does well across the entire movie. 
