import cv2.cv2 as cv2


class StaticRegionDetectors:

    @staticmethod
    def periorbital_region():
        pass

    @staticmethod
    def nose_region():
        pass

    @staticmethod
    def supraorbital_region():
        pass

    @staticmethod
    def forehead_region():
        pass

    @staticmethod
    def maxillary_region():
        pass


class FaceRegion:
    index = 1

    def __init__(self, name, landmarks):
        print(self.index)
        self.index = FaceRegion.index
        FaceRegion.index += 1

    def get_mean_std_temperature(self, thermal_image):
        pass


class BoundaryOpenCV:
    SHAPE_TYPES = ['line', 'circle', 'ellipse', 'rectangle']

    def __init__(self, shape_type, dict_parameters):
        if shape_type not in BoundaryOpenCV.SHAPE_TYPES:
            raise ValueError('Incorrect type shape: {}'.format(shape_type))

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
        keys = params.keys
        assert all(key in keys for key in ['pt1', 'pt2', 'color'])

        cv2.line(image, pt1=params['pt1'], pt2=params['pt2'], color=params['color'],
                 thickness=params.get('thickness'),
                 lineType=params.get('lineType'),
                 shift=params.get('shift'))

    def __draw_circle(self, image):
        params = self.dict_parameters
        keys = params.keys
        assert all(key in keys for key in ['center', 'radius', 'color'])

        cv2.circle(image, center=params['center'], radius=params['radius'], color=params['color'],
                   thickness=params.get('thickness'),
                   lineType=params.get('lineType'),
                   shift=params.get('shift'))

    def __draw_ellipse(self, image):
        params = self.dict_parameters
        keys = params.keys
        assert all(key in keys for key in ['center', 'axes', 'angle', 'startAngle', 'endAngle', 'color'])

        cv2.ellipse(image, center=params['center'], axes=params['axes'], angle=params['angle'],
                    startAngle=params['startAngle'],
                    endAngle=params['endAngle'],
                    color=params['color'],
                    thickness=params.get('thickness'),
                    lineType=params.get('lineType'),
                    shift=params.get('shift'))

    def __draw_rectangle(self, image):
        params = self.dict_parameters
        keys = params.keys
        assert all(key in keys for key in ['pt1', 'pt2', 'color'])

        cv2.rectangle(image, pt1=params['pt1'], pt2=params['pt2'], color=params['color'],
                      thickness=params.get('thickness'),
                      lineType=params.get('lineType'),
                      shift=params.get('shift'))
