# AR Tag Detection
[![Build Status](https://travis-ci.org/urastogi885/ar-tag-detection.svg?branch=master)](https://travis-ci.org/urastogi885/ar-tag-detection)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/ar-tag-detection/blob/master/LICENSE)

Detection of AR tags without using OpenCV homography functions

## Overview

Detection of AR tags without using OpenCV homography functions


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
import numpy as np
````

## Run

- Extract the compressed folder onto your system
- Go into the *Code* sub-directory
- Make sure all dependencies have been installed and all the files such as *Lena.png* and *Tag0.mp4* have been added to
the *Code* sub-directory.
- Open a terminal window by right-clicking on empty space within the folder and then click ````Open in Terminal````
- Run main program:
````
python3 main.py
````