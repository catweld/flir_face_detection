from flir_image_extractor import FlirImageExtractor
from image_processing import ImageProcessor, StreamProcessor

import cv2.cv2 as cv2
import tiffcapture as tc


def main_webcam_stream():
    image_processor = ImageProcessor(option='dlib_68landmarks')
    stream_processor = StreamProcessor(image_processor, index_cam=2)

    stream_processor.run()


def main_flir_image_processor():
    """
    test filenames
    flir_20190617T163823.jpg
    flir_20190618T092717.jpg
    flir_20190618T143825.jpg
    flir_20190619T161841.jpg
    flir_20190619T161856.jpg
    flir_20190619T161858.jpg
    flir_20190620T113530.jpg (horizontal)
    flir_20190620T113555.jpg (horizontal)
    FLIROne-2019-07-09-14-47-31+0200.jpg
    """
    input_file = 'test_images/' + 'FLIROne-2019-07-09-14-47-31+0200.jpg'

    fie = FlirImageExtractor()
    fie.process_image(input_file, upsample_thermal=True, transform_rgb=True)

    # fie.plot()

    rgb_image = fie.get_rgb_np()
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    thermal_image_3d = fie.img_thermal_rgb
    thermal_image_3d = cv2.cvtColor(thermal_image_3d, cv2.COLOR_BGR2RGB)

    # Creating region contours
    image_processor = ImageProcessor(option='dlib_68landmarks')
    image_processor.process_image(rgb_image)

    cv2.imshow("RGB image with contours", rgb_image)
    # cv2.waitKey(0)

    image_processor.apply_saved_contours(thermal_image_3d)

    cv2.imshow("Thermal image with contours", thermal_image_3d)
    cv2.waitKey(0)

    thermal_image_raw = fie.get_thermal_np()
    regions = image_processor.all_regions
    print('Region\tMean T\tStd T')

    for region in regions:
        mean, std = region.get_mean_std_temperature(thermal_image_raw)
        print('{}\t{}\t{}'.format(region.name, mean, std))
    # fie.export_thermal_to_csv('thermals_csv/'+file_name+'_thermal_csv.csv')

    fie.save_images()


def main_flir_video_processor():
    input_file = 'test_videos/flir_20190624T120020.mp4'

    cap = cv2.VideoCapture(input_file)

    # Check if camera opened successfully
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    fie = FlirImageExtractor()

    count = 0
    # Read until video is completed
    while cap.isOpened():
        # cap.set(cv2.CAP_PROP_CONVERT_RGB, cv2.CV_16U)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            tmp_frame_filename = "tmp_frames/frame%d.tif" % count
            cv2.imwrite(tmp_frame_filename, frame)
            count += 1

            # Processing image
            fie.process_image(tmp_frame_filename, upsample_thermal=True, transform_rgb=True)

            rgb_image = fie.get_rgb_np()
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            thermal_image_3d = fie.img_thermal_rgb
            thermal_image_3d = cv2.cvtColor(thermal_image_3d, cv2.COLOR_BGR2RGB)

            # Creating region contours
            image_processor = ImageProcessor(option='dlib_68landmarks')
            image_processor.process_image(rgb_image)

            cv2.imshow("RGB image with contours", rgb_image)
            # cv2.waitKey(0)

            image_processor.apply_saved_contours(thermal_image_3d)

            cv2.imshow("Thermal image with contours", thermal_image_3d)
            cv2.waitKey(0)

            thermal_image_raw = fie.get_thermal_np()
            regions = image_processor.all_regions
            print('Region\tMean T\tStd T')

            for region in regions:
                mean, std = region.get_mean_std_temperature(thermal_image_raw)
                print('{}\t{}\t{}'.format(region.name, mean, std))

            # Press Q on keyboard to  exit
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break


        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def main_server_flir_app():
    import socket
    import numpy

    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    TCP_IP = socket.gethostname()
    TCP_PORT = 1234
    server_address = (TCP_IP, TCP_PORT)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.bind(server_address)
    except Exception as ex:
        print("Not connected: " + str(ex))

    print(s)
    s.listen(True)
    s.settimeout(60)

    stopped = False
    while not stopped:
        try:
            conn, (ip, port) = s.accept()

            length = recvall(conn, 4)
            if length is None:
                continue
            length = int.from_bytes(length, "big")
            if not length > 0:
                continue
            stringData = recvall(conn, length)
            # data = numpy.fromstring(stringData, dtype='uint8')
            data = numpy.frombuffer(stringData, dtype='uint8')

            decimg = cv2.imdecode(data, 1)
            cv2.imshow('SERVER', decimg)
            cv2.waitKey(1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        except socket.timeout:
            print("Socket timed out")
            stopped = True
    s.close()


if __name__ == '__main__':
    # main_webcam_stream()
    # main_flir_image_processor()
    # main_flir_video_processor()
    main_server_flir_app()
