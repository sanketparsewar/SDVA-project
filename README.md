# Social Distancing Violation Analyzer

## Overview

The <b> Social Distancing Violation Analyzer </b> is a project that utilizes the YOLOv3 object detection algorithm and the COCO dataset to analyze social distancing violations in images or videos. The goal of this project is to detect and track individuals in a given scene and determine if they are maintaining a safe distance from each other.


## Features

<ul>
  <li>Object detection: The project utilizes the YOLOv3 algorithm to detect and track individuals in a given image or video</li>
  <li>Social distancing violation analysis: The analyzer calculates the distance between individuals and determines if they are violating the </li>
  <li>recommended social distancing guidelines.</li>
  <li>Using the centroid of the boxes we then measure the distances between them.</li>
  <li>For the distance measure, <b>Euclidean Distance</b> was used.</li>
  <li>Visualization: The violations are visualized by drawing bounding boxes around individuals and labeling them as safe or violating.</li>
  <li>A box is colored RED if unsafe and GREEN if safe.</li>
</ul>

## Requirements
Python 3.x
OpenCV
NumPy
sQlite3

## Getting started

1. Clone and download the repo
```bash
  git clone https://github.com/sanketparsewar/SDVA-project 
```

2. Then download the YOLOv3 weights from this <a href="https://pjreddie.com/media/files/yolov3.weights">link</a>

3. Download the required python packages
```bash
pip install -r requirements.txt
```

4. Run the app.py file
```bash
python app.py
```

## Demo

demo video

![](demoVideo/output-demo.mp4)

## Limitations and Future Scope

<ul>
  <li>This project does not take into account the camera perspective.</li>
  <li>It does not leverage a proper camera calibration (Distances are not measure accurate).</li>
</ul>

&nbsp;<b><i>Will work on these limitations.</i><b>

## References

<ul>
  <li><a href="https://www.pyimagesearch.com/start-here/">Getting started with OpenCV</a></li>
  <li><a href="https://pjreddie.com/darknet/yolo/">YOLO for Object Detection</a></li>
</ul>

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License.

