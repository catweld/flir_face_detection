from flir_image_extractor import FlirImageExtractor
from image_processing import ImageProcessor, StreamProcessor

import cv2.cv2 as cv2


def main_webcam_stream():
    image_processor = ImageProcessor(option='dlib_68landmarks')
    stream_processor = StreamProcessor(image_processor, index_cam=0)

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
    """
    input_file = 'test_images/flir_20190620T113530.jpg'

    fie = FlirImageExtractor()
    fie.process_image(input_file, upsample_thermal=True, transform_rgb=True)

    # fie.plot()

    rgb_image = fie.get_rgb_np()
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    thermal_image_3d = fie.img_thermal_rgb
    thermal_image_3d = cv2.cvtColor(thermal_image_3d, cv2.COLOR_BGR2RGB)

    # Creating region boundaries
    image_processor = ImageProcessor(option='dlib_68landmarks')
    image_processor.process_image(rgb_image)

    cv2.imshow("RGB image with boundaries", rgb_image)
    # cv2.waitKey(0)

    image_processor.apply_saved_boundaries(thermal_image_3d)

    cv2.imshow("Thermal image with boundaries", thermal_image_3d)
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
    pass


if __name__ == '__main__':
    # main_webcam_stream()
    main_flir_image_processor()
