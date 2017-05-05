**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[example]: ./output_images/image_example.jpg
[hog0]: ./output_images/ch0_and_hog.jpg
[hog1]: ./output_images/ch1_and_hog.jpg
[hog2]: ./output_images/ch2_and_hog.jpg
[sliding_window]: ./output_images/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_result.mp4

Below are the codes, video and images related to in this write up.
* Codes
  1. `lesson_functions.py`
    Some helper functions are defined in this file. All codes are almost identical to the codes presented in Udacity lesson. This file is imported in the following 2 ipython notebook.
  1. `train_classifier.ipynb`
    Here, SVM classifier was trained using dataset images. The classifier is saved as a pickle file named "svc.pickle".
  1. `project_notebook.ipynb`
    This is where the project video was actually processed. Classifier trained at train_classifier.ipynb was used.
* Videos
  1. `project_video.mp4`
    Original video before processing.
  1. `project_video_result.mp4`
    Final output of this project, which contains bounding boxes that detect cars and corresponding (translucent) heat map.
* Images
  1. `dataset/`
    Images that were used for training SVM classifier. Provided by Udacity.
  1. `output_images/`
    Images used for explanation in this write up.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell @ train_classifier.ipynb.

I started by reading in all the `vehicle` and `non-vehicle` images (2nd code cell @ train_classifier.ipynb). Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][example]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb`
 color space and OG param1ters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`2


![alt text][hog0]
![alt text][hog1]
![alt text][hog2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and chose one that have shown highest test accuracy. As a result I have chosen the following parameters' combination.

| Color space | YCrCb |
| HOG orientation | 8 |
| HOG pixels per cell | (8,8) |
| HOG cells per block | (2,2) |
| Spatial binning dimensions | (32,32) |
| Number of histogram bins | 16 |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using scikit-learn library, which is written in 4th code cell @ train_classifier.ipynb. Training data were randomized to eliminate the effect of orderingm and splited to training and test set at a ratio of 80% and 20%. `StandardScaler()` was used to standardize the feature values which consist of different kind of feature: HOG, Spatial and Histogram features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions between 400-656 pixels in y-axis, which has potential of car exisistence, and at 3 kind of scales: 64*64, 92*92 and 128*128 pixels. The codes are in 3rd code cell @ project_notebook.ipynb.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][sliding_window]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a (thresolded) temporial heatmap and it was passed through low pass filter to eliminate noise (or outliers) and to obtain smooth detection result.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Related codes are in 3rd code cell @ project_notebook.ipynb.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. In the video, the implementation momentarily failed to detect white car @ 29sec. This is because the classifier couldn't detect the car and also partly due to the setting of low pass filter. To gain better result, the program should hold similar heat map for longer time after the program has high confidence on car's exisistence even it temporarily lost the detection.

1. In the video, when a car was overlapped by the other car, the program only detected the 2 cars as a single car. In order to overcome the problem, not only using heat map for detection, some codes which stores and estimates individual cars' location should be implemented.
