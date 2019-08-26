from flir_image_extractor import FlirImageExtractor
from image_processing import ImageProcessor, StreamProcessor

from PIL import Image, ImageFile
import cv2.cv2 as cv2


ImageFile.LOAD_TRUNCATED_IMAGES = True


def main_webcam_stream():
    """
    Apply a model on a stream of data coming from a connected cam.
    """
    image_processor = ImageProcessor(option='dlib_68landmarks')
    stream_processor = StreamProcessor(image_processor, index_cam=0)

    stream_processor.run()


def main_flir_image_processor():
    """
    Process an image taken with a Flir One Pro thermal camera.
    The image can be taken either with the official app or a custom app using the Flir's SDK.
    This does not work on videos splitted frame by frame, because the camera doesn't save
    thermal information when recording a video.

    I have left a picture to test out the script.
    """
    input_file = 'test_images/' + 'flir_20190617T163823.jpg'

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

    # thermal_image_raw = fie.get_thermal_np()
    # regions = image_processor.all_regions
    # print('Region\tMean T\tStd T')
    #
    # for region in regions:
    #     mean, std = region.get_mean_std_temperature(thermal_image_raw)
    #     print('{}\t{}\t{}'.format(region.name, mean, std))

    # fie.export_thermal_to_csv('thermals_csv/'+file_name+'_thermal_csv.csv')

    fie.save_images()


def main_server_flir_app():
    """
    Start displaying the images received by the Flir device as soon as they are available.
    """
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
    s.settimeout(60000)

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
    """
    It is possible to choose between 3 different applications.
    """
    main_webcam_stream()
    # main_flir_image_processor()
    # main_server_flir_app()
