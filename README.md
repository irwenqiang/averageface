Average Face
===========

Python tool for creating average images from faces


**new features in @mckelvin 's fork version:**

- add cache for features detecting
- both eyes and nose of a face are aligned for better image quality
- more robust when handling large amount of pictures


Requirements
------------

- OpenCV /w Python bindings (`yaourt -S opencv` for archlinux)
- numpy
- PIL


Usage
-----

    python averageface.py "/tmp/faces/*.jpg" output_average_face.jpg
