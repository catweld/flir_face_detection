import cv2.cv2 as cv2

from math import ceil, sqrt


class StaticBoundariesDetectors:

    @staticmethod
    def periorbital_boundaries(landmarks):
        center_periorbital_left = ((landmarks.part(27).x + landmarks.part(39).x) // 2,
                                   (landmarks.part(27).y + landmarks.part(39).y) // 2)

        radius_periorbital_left = ceil(sqrt((landmarks.part(27).x - landmarks.part(39).x) ** 2
                                            + (landmarks.part(27).y - landmarks.part(39).y) ** 2)) // 2

        center_periorbital_right = ((landmarks.part(42).x + landmarks.part(27).x) // 2,
                                    (landmarks.part(42).y + landmarks.part(27).y) // 2)

        radius_periorbital_right = ceil(sqrt((landmarks.part(42).x - landmarks.part(27).x) ** 2
                                             + (landmarks.part(42).y - landmarks.part(27).y) ** 2)) // 2

        periorbital_left = BoundaryOpenCV('circle',
                                          dict(center=(center_periorbital_left[0], center_periorbital_left[1]),
                                               radius=radius_periorbital_left,
                                               color=(255, 0, 0),
                                               lineType=8))
        periorbital_right = BoundaryOpenCV('circle',
                                           dict(center=(center_periorbital_right[0], center_periorbital_right[1]),
                                                radius=radius_periorbital_right,
                                                color=(255, 0, 0),
                                                lineType=8))

        return [periorbital_left, periorbital_right]

    @staticmethod
    def nose_boundaries(landmarks):
        pass

    @staticmethod
    def supraorbital_boundaries(landmarks):
        pass

    @staticmethod
    def forehead_boundaries(landmarks):
        pass

    @staticmethod
    def maxillary_boundaries(landmarks):
        pass


class FaceRegion:
    index = 1

    def __init__(self, name, detector_method):
        self.index = FaceRegion.index
        FaceRegion.index += 1
        self.name = name

        self.boundaries = None

        self.detector = detector_method

    def detect_region(self, landmarks):
        self.boundaries = self.detector(landmarks)

    def get_mean_std_temperature(self, thermal_image):
        pass


class BoundaryOpenCV:
    SHAPE_TYPES = ['line', 'circle', 'ellipse', 'rectangle']

    def __init__(self, shape_type, dict_parameters):
        if shape_type not in BoundaryOpenCV.SHAPE_TYPES:
            raise ValueError('Incorrect type shape: {}'.format(shape_type))

        self.shape_type = shape_type
        self.__draw_shape = None
        self.dict_parameters = dict_parameters

        if shape_type == 'line':
            self.__draw_shape = self.__draw_line
        elif shape_type == 'circle':
            self.__draw_shape = self.__draw_circle
        elif shape_type == 'ellipse':
            self.__draw_shape = self.__draw_ellipse
        elif shape_type == 'rectangle':
            self.__draw_shape = self.__draw_rectangle

    def apply_to_image(self, image):
        """
        Draws the shape on the image.
        :param image: RGB image as numpy array
        """
        self.__draw_shape(image)

    def __draw_line(self, image):
        params = self.dict_parameters
        keys = params.keys()
        assert all(key in keys for key in ['pt1', 'pt2', 'color'])

        cv2.line(image, pt1=params['pt1'], pt2=params['pt2'], color=params['color'],
                 thickness=params.get('thickness') or 1,
                 lineType=params.get('lineType') or 8,
                 shift=params.get('shift') or 0)

    def __draw_circle(self, image):
        params = self.dict_parameters
        keys = params.keys()
        assert all(key in keys for key in ['center', 'radius', 'color'])

        cv2.circle(image, center=params['center'], radius=params['radius'], color=params['color'],
                   thickness=params.get('thickness') or 1,
                   lineType=params.get('lineType') or 8,
                   shift=params.get('shift') or 0)

    def __draw_ellipse(self, image):
        params = self.dict_parameters
        keys = params.keys()
        assert all(key in keys for key in ['center', 'axes', 'angle', 'startAngle', 'endAngle', 'color'])

        cv2.ellipse(image, center=params['center'], axes=params['axes'], angle=params['angle'],
                    startAngle=params['startAngle'],
                    endAngle=params['endAngle'],
                    color=params['color'],
                    thickness=params.get('thickness') or 1,
                    lineType=params.get('lineType') or 8,
                    shift=params.get('shift') or 0)

    def __draw_rectangle(self, image):
        params = self.dict_parameters
        keys = params.keys()
        assert all(key in keys for key in ['pt1', 'pt2', 'color'])

        cv2.rectangle(image, pt1=params['pt1'], pt2=params['pt2'], color=params['color'],
                      thickness=params.get('thickness') or 1,
                      lineType=params.get('lineType') or 8,
                      shift=params.get('shift') or 0)
