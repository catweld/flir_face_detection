from flir_image_extractor import FlirImageExtractor
from image_processing import ImageProcessor, StreamProcessor


def main_webcam_stream():
    image_processor = ImageProcessor(option='dlib_68landmarks')
    stream_processor = StreamProcessor(image_processor, index_cam=0)

    stream_processor.run()


def main_flir_image_processor():
    input_file = 'test_images/flir_20190618T092717.jpg'
    file_name = input_file.split('/')[1].split('.')[0]

    fie = FlirImageExtractor()
    fie.process_image(input_file)

    # fie.plot()

    # fie.export_thermal_to_csv('thermals_csv/'+file_name+'_thermal_csv.csv')

    fie.save_images()


def main_flir_video_processor():
    pass


if __name__ == '__main__':
    main_webcam_stream()
    # main_flir_image_processor()
