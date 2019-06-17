import cv2
import sys
import os


class ImageProcessor:
    def __init__(self, option='faces', path_cv2_models=None):

        self.model_fullpath = None
        self.model = None
        self.path_cv2_models = os.getcwd() + '\\flir_env\\Lib\\site-packages\\cv2\\data\\' if path_cv2_models is None else path_cv2_models
        self.__processor = self.__select_processor(option)

    def __select_processor(self, option):
        POSSIBLE_OPTIONS = ['faces']

        if option not in POSSIBLE_OPTIONS:
            raise ValueError('\'{}\' is not a valid option ({})'.format(option, POSSIBLE_OPTIONS))

        if option == 'faces':
            self.model_fullpath = self.path_cv2_models + 'haarcascade_frontalface_default.xml'
            self.model = cv2.CascadeClassifier(self.model_fullpath)
            return self.__detect_and_draw_faces

    def __detect_and_draw_faces(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def process_image(self, image):
        self.__processor(image)

class StreamProcessor:

    def __init__(self, image_processor, index_cam=0):
        """
        Parameters:
             index_cam: Integer defining which cam to use for the input stream.
                If not provided, the first cam found will be used.
                Otherwise, an exception will be raised.
        """
        self.index_cam = index_cam
        self.capture_stream = self.__init_capture_stream()
        self.image_processor = image_processor

    def __init_capture_stream(self):

        if self.index_cam > 0:
            cap = self.__test_webcam(self.index_cam)
            if cap is None:
                print('First webcam will be used instead.', file=sys.stderr)
                self.index_cam = 0

        cap = self.__test_webcam(self.index_cam)
        if cap is None:
            print('Error with first webcam. Please, connect a webcam.', file=sys.stderr)
            exit()

        print('Webcam connected')
        return cap

    def __test_webcam(self, index):
        try:
            cap = cv2.VideoCapture(self.index_cam)
            ret, img = cap.read()
            cv2.imshow("input", img)
        except cv2.error:
            print('Webcam {} not found'.format(index), file=sys.stderr)
            cap = None

        return cap

    def run(self):
        while True:
            # start = datetime.datetime.now()

            ret, img = self.capture_stream.read()

            self.image_processor.process_image(img)
            cv2.imshow("input", img)
            # end = datetime.datetime.now()

            # ms = (end-start).microseconds/1000
            # hz = int(1000//ms) if ms > 0 else 0
            # print('Time: {:.2f}ms\tHz: {:d}'.format(ms, hz))

            key = cv2.waitKey(10)
            if key == 27:
                break

        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
