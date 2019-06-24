import cv2.cv2 as cv2

from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt


class StaticContoursDetectors:

    @staticmethod
    def periorbital_contours(landmarks):
        center_periorbital_left = ((landmarks.part(27).x + landmarks.part(39).x) // 2,
                                   (landmarks.part(27).y + landmarks.part(39).y) // 2)

        radius_periorbital_left = ceil(sqrt((landmarks.part(27).x - landmarks.part(39).x) ** 2
                                            + (landmarks.part(27).y - landmarks.part(39).y) ** 2)) // 2

        center_periorbital_right = ((landmarks.part(42).x + landmarks.part(27).x) // 2,
                                    (landmarks.part(42).y + landmarks.part(27).y) // 2)

        radius_periorbital_right = ceil(sqrt((landmarks.part(42).x - landmarks.part(27).x) ** 2
                                             + (landmarks.part(42).y - landmarks.part(27).y) ** 2)) // 2

        periorbital_left = ContourOpenCV('circle',
                                         dict(center=(center_periorbital_left[0], center_periorbital_left[1]),
                                              radius=radius_periorbital_left,
                                              color=(255, 0, 0),
                                              lineType=8))
        periorbital_right = ContourOpenCV('circle',
                                          dict(center=(center_periorbital_right[0], center_periorbital_right[1]),
                                               radius=radius_periorbital_right,
                                               color=(255, 0, 0),
                                               lineType=8))

        return [periorbital_left, periorbital_right]

    @staticmethod
    def nose_contours(landmarks):
        pass

    @staticmethod
    def supraorbital_contours(landmarks):
        pass

    @staticmethod
    def forehead_contours(landmarks):
        pass

    @staticmethod
    def maxillary_contours(landmarks):
        pass


class FaceRegion:
    index = 1

    def __init__(self, name, detector_method):
        self.index = FaceRegion.index
        FaceRegion.index += 1
        self.name = name

        self.contours = None

        self.detector = detector_method

        self.region_mask = None
        self.region_points = None

    def detect_region(self, image, landmarks):
        self.contours = self.detector(landmarks)
        # Code from https://stackoverflow.com/questions/14083256/retrieve-circle-points
        mask = np.zeros(image.shape[:2], dtype="uint8")

        for contour in self.contours:
            contour.apply_to_image(mask, filled=True)

        self.region_mask = mask
        self.region_points = np.transpose(np.where(mask == 255))

    def get_mean_std_temperature(self, thermal_image):
        plt.imshow(self.region_mask)
        plt.show()
        mean, std = cv2.meanStdDev(thermal_image, mask=self.region_mask)
        return mean.item(), std.item()


class ContourOpenCV:
    SHAPE_TYPES = ['line', 'circle', 'ellipse', 'rectangle']

    def __init__(self, shape_type, dict_parameters):
        if shape_type not in ContourOpenCV.SHAPE_TYPES:
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

    def apply_to_image(self, image, filled=False, translation=(0, 0)):
        """
        Draws the shape on the image.
        :param image: RGB image as numpy array
        """
        self.__draw_shape(image, filled, translation)

    def __draw_line(self, image):
        params = self.dict_parameters
        keys = params.keys()
        assert all(key in keys for key in ['pt1', 'pt2', 'color'])

        cv2.line(image, pt1=params['pt1'], pt2=params['pt2'], color=params['color'],
                 thickness=params.get('thickness') or 1,
                 lineType=params.get('lineType') or 8,
                 shift=params.get('shift') or 0)

    def __draw_circle(self, image, filled, translation):
        params = self.dict_parameters
        keys = params.keys()
        assert all(key in keys for key in ['center', 'radius', 'color'])

        if filled:
            color = 255
            thickness = -1
        else:
            color = params['color']
            thickness = params.get('thickness') or 1

        center = (params['center'][0] + translation[0], params['center'][1] + translation[1])

        cv2.circle(image, center=center, radius=params['radius'], color=color,
                   thickness=thickness,
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
