import cv2
import sys
import os


class ImageProcessor:
    def __init__(self, option='faces', path_cv2_models=None):

        self.models = {}
        self.path_cv2_models = os.getcwd() + '\\flir_env\\Lib\\site-packages\\cv2\\data\\' if path_cv2_models is None else path_cv2_models
        self.__processor = self.__select_processor(option)

    def __select_processor(self, option):
        POSSIBLE_OPTIONS = ['faces', 'faces_and_eyes', 'better_faces_and_eyes']

        if option not in POSSIBLE_OPTIONS:
            raise ValueError('\'{}\' is not a valid option ({})'.format(option, POSSIBLE_OPTIONS))

        if option == 'faces':
            self.models['faces'] = cv2.CascadeClassifier(self.path_cv2_models + 'haarcascade_frontalface_default.xml')
            return self.__detect_and_draw_faces
        elif option == 'faces_and_eyes':
            self.models['faces'] = cv2.CascadeClassifier(self.path_cv2_models + 'haarcascade_frontalface_default.xml')
            self.models['eyes'] = cv2.CascadeClassifier(self.path_cv2_models + 'haarcascade_eye.xml')
            return self.__detect_and_draw_faces_and_eyes
        elif option == 'better_faces_and_eyes':
            self.models['faces'] = cv2.CascadeClassifier(self.path_cv2_models + 'haarcascade_frontalface_default.xml')
            self.models['eyes'] = cv2.CascadeClassifier(self.path_cv2_models + 'haarcascade_eye.xml')
            return self.__detect_and_draw_faces_and_eyes_better


    def __detect_faces(self, gray_frame):

        # Detect faces
        faces = self.models['faces'].detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return faces

    def __draw_faces(self, faces, frame):
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def __detect_eyes(self, gray_frame):
        eyes = self.models['eyes'].detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return eyes

    def __draw_eyes(self, eyes, frame):
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    def __detect_and_draw_faces_and_eyes(self, frame, gray_frame):
        faces = self.__detect_faces(gray_frame)
        self.__draw_faces(faces, frame)

        eyes = self.__detect_eyes(gray_frame)
        self.__draw_eyes(eyes, frame)

    def __detect_and_draw_faces_and_eyes_better(self, frame, gray_frame):
        pass

    def process_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.__processor(image, gray)


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
