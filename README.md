# AR Tag Detection
[![Build Status](https://travis-ci.org/urastogi885/ar-tag-detection.svg?branch=master)](https://travis-ci.org/urastogi885/ar-tag-detection)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/urastogi885/ar-tag-detection/blob/master/LICENSE)

## Content

- [Overview](#overview)
    - [Tag Detection](#tag-detection)
    - [Image Superimposition](#image-superimposition)
    - [3D Projecttion](#3d-projection)
- [License](#license)
- [Team Members](#team-members)
- [Dependencies](#dependencies)
- [Install Dependencies](#instatll-dependencies)
- [Run](#run)
- [Output](#output)

## Overview

- The project is divided into 3 parts: AR-Tag detection, superimposition of an image onto to the tag, and drawing a cuboid
onto the tag. 
- OpenCV methods such as find-homography and warp-perspective have not been used to develop any part of the
project.
- The input and output of the project are video files.
- Ideally, all video formats that are supported by the OpenCV library should be supported by the project. We have only
tested the code with the MP4 format.
- Output of the project is a video, in AVI format, that combines all the 3 parts of the project.

## Tag Detection

- OpenCV methods such as [*findContours*](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html),
 [*contourArea*](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html), and
 [*approxPolyDP*](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html)
 have been used to detect contours within an image.
- Among the various contours found from the image, ar-tags were found by restricting our search to contours with closed
regions of 4 points.
- The output can be seen below in *Figure 1* for single as well as multiple tags.

<p align="center">
  <img src="https://github.com/urastogi885/ar-tag-detection/blob/master/Code/images/ar_tag_detection.png">
  <b>Figure 1 - Stages of tag detection</b>
</p>

### Image Superimposition

- Methods to find homography and warp images were developed to warp the image in *Figure 2* into the video frame.
- The final output can be seen in *Figure 3*.

<p align="center">
  <img src="https://github.com/urastogi885/ar-tag-detection/blob/master/Code/images/Lena.png">
  <br><b>Figure 2 - Image for superimposition</b><br>
</p>
<p align="center">
  <img src="https://github.com/urastogi885/ar-tag-detection/blob/master/Code/images/image_superimposition.png">
  <b>Figure 3 - Superimposition of image on single and multiple AR Tags</b>
</p>

### 3D Projection

- Reference 3D points were taken, then new points were found using [*projectionPoints*](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#projectpoints)
method of OpenCV and finally [*drawContours*](https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours)
was used to draw the cube on the video frame.
- 3D projection can be seen in *Figure 4*.

<p align="center">
  <img src="https://github.com/urastogi885/ar-tag-detection/blob/master/Code/images/3d_projection.png">
  <b>Figure 4 - 3D-Projection of cuboid on single and multiple AR Tags</b>
</p>

## [License](https://github.com/urastogi885/ar-tag-detection/blob/master/LICENSE)

- The project is developed under the BSD 3-Clause license to comply with the standard license of the OpenCV library.

## Team Members

- [Umang Rastogi](https://www.linkedin.com/in/urastogi96/) - Robotics graduate student at UMD interested in working in
the autonomous vehicle industry
- [Sayani Roy](https://www.linkedin.com/in/roysayani/) - UMD Robotics Graduate student interested in medical 
robotics
- [Prateek Bhargava](https://www.linkedin.com/in/prateek96/) - UMD Robotics Graduate student interested in space 
robotics


## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: Opencv-python and Numpy

## Install Dependencies

- Install *Python3*, *Python3-tk*, and the necessary libraries: (if not already installed)
````
sudo apt install python3 python3-tk
pip3 install opencv-python numpy
````
- Check if your system successfully installed all the dependencies
- Open terminal using ````Ctrl+Alt+T```` and enter ````python3````
- The terminal should now present a new area represented by ````>>>```` to enter python commands
- Now use the following commands to check libraries: (Exit python window using ````Ctrl+Z```` if an error pops up while 
running the below commands)
````
import tkinter
import cv2
import numpy
````

## Run

- Extract the compressed folder onto your system
- Go into the project directory
- Open a terminal window by right-clicking on empty space within the folder and then click ````Open in Terminal````
and run the following commands:
````
cd Code/
python3 main.py path/input-file-name.mp4 path/output-file-name.avi
````
- For instance:
````
python3 main.py videos/Tag0.mp4 videos/output_tag0.avi
````
- Note that the program takes 2 input arguments to run: location of the input video file and destination to store the output video

## Output

- Checkout this [link](https://drive.google.com/drive/folders/1fPg8qZ5UhrjwJsX3OKZ2b5rX3MXvWC4G?usp=sharing) to go to
the google drive folder with all our final video outputs.
- Please note that each video contains output for all three parts of the project, i.e., tag detection, image
superimposition, and draw 3D cube on the tag.
- Output has been generated for each provided file, i.e., from *Tag0.mp4* to *multipleTags.mp4*.
- A sample of the final output can be seen in *Figure 5*.

 <p align="center">
  <img src="https://github.com/urastogi885/ar-tag-detection/blob/master/Code/images/final_output.png">
  <b>Figure 5 - Final output on multiple AR Tags</b>
</p>
