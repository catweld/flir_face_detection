import sys
import os
import dlib
from geometer import Point, Line
import cv2.cv2 as cv2
from regions import StaticContoursDetectors, ContourOpenCV, FaceRegion


class ImageProcessor:
    def __init__(self, option='faces', path_cv2_models=None, path_dlib_models=None):
        """
        :param option:
        :param path_cv2_models:
        :param path_dlib_models:
        """
        self.models = {}
        self.dlib_detector = None
        self.dlib_predictor = None
        self.path_cv2_models = os.getcwd() + '\\flir_env\\Lib\\site-packages\\cv2\\data\\' if path_cv2_models is None else path_cv2_models
        self.path_dlib_models = os.getcwd() + '\\dlib_models\\' if path_dlib_models is None else path_dlib_models
        self.__processor = self.__select_processor(option)
        self.contours = None
        self.__reset_contours()
        self.orientation = None
        self.all_regions = None

    def __select_processor(self, option):
        POSSIBLE_OPTIONS = ['faces', 'faces_and_eyes', 'better_faces_and_eyes', 'dlib_68landmarks']

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
        elif option == 'dlib_68landmarks':
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(self.path_dlib_models + "shape_predictor_68_face_landmarks.dat")
            return self.__detect_and_draw_dlib_landmarks

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

    def __detect_and_draw_faces(self, frame, gray_frame):
        faces = self.__detect_faces(gray_frame)
        self.__draw_faces(faces, frame)

    def __detect_eyes(self, gray_frame):
        eyes = self.models['eyes'].detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=15,
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
        faces = self.__detect_faces(gray_frame)
        # self.__draw_faces(faces, frame)

        if len(faces) == 0:
            # eyes = self.__detect_eyes(gray_frame)
            # self.__draw_eyes(eyes, frame)
            pass
        else:
            for face in faces:
                (x, y, w, h) = face
                roi_gray = gray_frame[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = self.__detect_eyes(roi_gray)

                if len(eyes) == 2:
                    self.__draw_faces([face], frame)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    eyes_centers = [(ex + ew // 2, ey + eh // 2) for (ex, ey, ew, eh) in eyes]

                    cv2.line(roi_color, (eyes_centers[0][0], eyes_centers[0][1]),
                             (eyes_centers[1][0], eyes_centers[1][1]), (66, 215, 244), 1)

                    eyes_centers_as_points = [Point(x, y) for (x, y) in eyes_centers]

                    middle_point_eyes = ((eyes_centers[0][0] + eyes_centers[1][0]) // 2,
                                         (eyes_centers[0][1] + eyes_centers[1][1]) // 2)

                    middle_point_eyes_as_point = Point(middle_point_eyes[0], middle_point_eyes[1])

                    line_centering_eyes = Line(eyes_centers_as_points[0], eyes_centers_as_points[1])

                    vertical_axis = line_centering_eyes.perpendicular(through=middle_point_eyes_as_point)

    def __detect_and_draw_dlib_landmarks(self, frame, gray_frame):
        faces = self.dlib_detector(gray_frame)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = self.dlib_predictor(gray_frame, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                # cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                self.contours['circle'].append(
                    ContourOpenCV('circle',
                                  dict(center=(x, y),
                                       radius=4,
                                       color=(255, 0, 0),
                                       thickness=-1)))

            # Detection and drawing of the regions
            periorbital_region = FaceRegion('Periorbital region', StaticContoursDetectors.periorbital_contours)
            periorbital_region.detect_region(frame, landmarks)
            self.all_regions.append(periorbital_region)
            self.__add_contours(periorbital_region.contours)

            forehead_region = FaceRegion('Forehead region', StaticContoursDetectors.forehead_contours)
            forehead_region.detect_region(frame, landmarks)
            self.all_regions.append(forehead_region)
            self.__add_contours(forehead_region.contours)

            for contour_type in self.contours.keys():
                for contour in self.contours[contour_type]:
                    contour.apply_to_image(frame)

    def process_image(self, image):
        self.all_regions = []
        self.__reset_contours()

        self.orientation = 'vertical' if image.shape[0] > image.shape[1] else 'horizontal'

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.__processor(image, gray)

    def apply_saved_contours(self, image):
        # Calibration for a vertical image
        if self.orientation == 'vertical':
            calibration = {
                'up_shift': 32,
                'right_shift': -14,
            }
        elif self.orientation == 'horizontal':
            calibration = {
                'up_shift': -35,
                'right_shift': 70,
            }
        else:
            raise ValueError('Orientation not valid.')

        for contour_type in self.contours.keys():
            for contour in self.contours[contour_type]:
                contour.apply_to_image(image, translation=(calibration['right_shift'], - calibration['up_shift']))

        # # Add circles
        # for circle in self.contours['circle']:
        #     circle.apply_to_image(image,
        #                           translation=(calibration['right_shift'], - calibration['up_shift']))

    def __reset_contours(self):
        self.contours = {'rectangle': [], 'line': [], 'circle': [], 'ellipse': [], 'poly': []}

    def __add_contours(self, contours):
        for contour in contours:
            self.contours[contour.shape_type].append(contour)


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
