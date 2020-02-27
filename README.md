# AR Tag Detection
[![Build Status](https://travis-ci.org/urastogi885/ar-tag-detection.svg?branch=master)](https://travis-ci.org/urastogi885/ar-tag-detection)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/ar-tag-detection/blob/master/LICENSE)

## Overview

Detection of AR tags without using *findHomography* and *warpPerspective* functions

## Team Members

- [Umang Rastogi](https://www.linkedin.com/in/urastogi96/) - Robotics graduate student at UMD interested in working in
the autonomous vehicle industry
- [Sayani Roy](https://www.linkedin.com/in/roysayani/) - 
- [Prateek Bhargava](https://www.linkedin.com/in/prateek96/) - UMD Robotics Graduate student interested in space 
robotics. 


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