# Image processing for face landmarking with Flir ONE Pro
###Things you should know about the camera

It has a RGB camera as well as a thermal camera. Because of their positioning on the device, 
calibration is needed to align the two images. The camera automatically perform a calibration when needed.

The two cameras have different resolution:
1) RGB resolution: 1440 x 1080
2) Thermal resolution: 160 x 120

However, the camera automatically upscales the thermal image to 640x480.
With this script, it is possible to upscale through interpolation to match the RGB resolution
by setting `upsample_thermal=True` when calling the `process_image()` function.
This is not enough to have a point (x,y) in the thermal image to correspond to the same point (x,y) in the RGB image.
The function `process_image()` has another parameter `transform_rgb` for making the two images match with a manual calibration.

### What this script does 

By running the different main methods, you can do three things:
1) Perform face landmarking and ROIs detection on images or stream of images
2) Process images taken with a Flir One Pro camera, find the contours of ROIs (Region Of Interest)
and compute mean and std of temperatures.
3) Live stream from a Flir One Pro Camera recording with this app https://gits-15.sys.kth.se/SoRoKTH/FLIRONEExampleApplication

####About 1 and 2
You can add the detection of new contours based on landmarks by adding a new method in the class `StaticContoursDetectors` in `regions.py`

Since the contours are defined with the landmarks, 
this code can easily be modified to use other models for the landmarks detection, for example OpenFace.
