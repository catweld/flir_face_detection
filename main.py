import argparse

from flir_image_extractor import FlirImageExtractor
import cv2
import datetime
from image_processing import ImageProcessor, StreamProcessor

def main_thermal_conversion():
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=True)
    parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
                        required=False)
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)
    fie.process_image(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()


def main_webcam_stream():

    image_processor = ImageProcessor(option='faces')
    stream_processor = StreamProcessor(image_processor, index_cam=0)

    stream_processor.run()

    # Change the camera setting using the set() function
    # cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, -6.0)
    # cap.set(cv2.cv.CV_CAP_PROP_GAIN, 4.0)
    # cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 144.0)
    # cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 27.0)
    # cap.set(cv2.cv.CV_CAP_PROP_HUE, 13.0) # 13.0
    # cap.set(cv2.cv.CV_CAP_PROP_SATURATION, 28.0)
    # Read the current setting from the camera

    # test = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
    # ratio = cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
    # frame_rate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    # width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    # brightness = cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
    # contrast = cap.get(cv2.cv.CV_CAP_PROP_CONTRAST)
    # saturation = cap.get(cv2.cv.CV_CAP_PROP_SATURATION)
    # hue = cap.get(cv2.cv.CV_CAP_PROP_HUE)
    # gain = cap.get(cv2.cv.CV_CAP_PROP_GAIN)
    # exposure = cap.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
    # print("Test: ", test)
    # print("Ratio: ", ratio)
    # print("Frame Rate: ", frame_rate)
    # print("Height: ", height)
    # print("Width: ", width)
    # print("Brightness: ", brightness)
    # print("Contrast: ", contrast)
    # print("Saturation: ", saturation)
    # print("Hue: ", hue)
    # print("Gain: ", gain)
    # print("Exposure: ", exposure)



    # while True:
    #     # start = datetime.datetime.now()
    #
    #     ret, img = cap.read()
    #
    #     detect_and_draw_faces(img)
    #     cv2.imshow("input", img)
    #     # end = datetime.datetime.now()
    #
    #     # ms = (end-start).microseconds/1000
    #     # hz = int(1000//ms) if ms > 0 else 0
    #     # print('Time: {:.2f}ms\tHz: {:d}'.format(ms, hz))
    #
    #     key = cv2.waitKey(10)
    #     if key == 27:
    #         break
    #
    #
    # cv2.destroyAllWindows()
    # cv2.VideoCapture(0).release()


# def detect_and_draw_faces(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE,
#     )
#
#     # Draw rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


if __name__ == '__main__':
    main_webcam_stream()
