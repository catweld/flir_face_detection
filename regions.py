import cv2.cv2 as cv2

from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt


class StaticContoursDetectors:
    # TODO: create constants for landmarks

    @staticmethod
    def periorbital_contours(landmarks):
        """
        Draw 2 circles based on the facial landmarks.
        One that passes through points 40 and 28, and with their distance as diameter
        Another one that passes through 28 and 43, and with their distance as diameter
        """

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
        top = (landmarks.part(27).x, landmarks.part(27).y)
        right_bottom = (landmarks.part(35).x, landmarks.part(35).y)
        left_bottom = (landmarks.part(31).x, landmarks.part(31).y)
        nose = ContourOpenCV('poly', dict(pts=np.array([top, right_bottom, left_bottom], dtype='int32'),
                                      color=(255, 0, 0),
                                      lineType=8))
        
        return [nose]

    @staticmethod
    def supraorbital_contours(landmarks):
        left_left = (landmarks.part(17).x, landmarks.part(17).y)
        left_center = (landmarks.part(19).x, landmarks.part(19).y)
        left_right = (landmarks.part(21).x, landmarks.part(21).y)
        
        right_left = (landmarks.part(22).x, landmarks.part(22).y)
        right_center = (landmarks.part(24).x, landmarks.part(24).y)
        right_right = (landmarks.part(26).x, landmarks.part(26).y)
        
#         supraorbital_left = ContourOpenCV('ellipse',
#                                           dict(center=(left_center[0], left_center[1]),
#                                                #radius=radius_periorbital_right,
#                                                color=(255, 0, 0),
#                                                lineType=8))

        supraorbital_left = ContourOpenCV('line', dict(pt1=np.array([left_left]), pt2=np.array([left_right]),
                                               color=(255, 0, 0),
                                               lineType=8))
        
#         supraorbital_right = ContourOpenCV('ellipse',
#                                           dict(center=(right_center[0], right_center[1]),
#                                                #radius=radius_periorbital_right,
#                                                color=(255, 0, 0),
#                                                lineType=8))

        supraorbital_right = ContourOpenCV('line', dict(pt1=np.array([right_left]), pt2=np.array([right_right]),
                                               color=(255, 0, 0),
                                               lineType=8))

        
        return [supraorbital_left, supraorbital_right]

    @staticmethod
    def forehead_contours(landmarks):
        """
        Draw a polygon with 4 vertices using landmarks 21 and 24.
        """
        # Maybe use an ellipse instead?

        left_i = 20
        right_i = 23
        down_left = (landmarks.part(left_i).x, landmarks.part(left_i).y)
        down_right = (landmarks.part(right_i).x, landmarks.part(right_i).y)

        width = ceil(sqrt((down_right[0] - down_left[0]) ** 2
                          + (down_right[1] - down_left[1]) ** 2))

        width_over_height = 1.6
        height = int(width // width_over_height)

        up_left = (down_left[0], down_left[1] - height)
        up_right = (down_right[0], down_right[1] - height)

        forehead = ContourOpenCV('poly',
                                 dict(pts=np.array([down_left, up_left, up_right, down_right], dtype='int32'),
                                      color=(255, 0, 0),
                                      lineType=8))
        return [forehead]
    
    @staticmethod
    def background_contours(landmarks):
        
        down_left = (10, 70)
        down_right = (300, 70)

        width = ceil(sqrt((down_right[0] - down_left[0]) ** 2
                          + (down_right[1] - down_left[1]) ** 2))

        width_over_height = 1.6
        height = int(width // width_over_height)

        up_left = (down_left[0], down_left[1] - height)
        up_right = (down_right[0], down_right[1] - height)

        background = ContourOpenCV('poly',
                                 dict(pts=np.array([down_left, up_left, up_right, down_right], dtype='int32'),
                                      color=(255, 0, 0),
                                      lineType=8))

#         background = ContourOpenCV('background',dict(pts=np.array([down_left, up_left, up_right, down_right], dtype='int32'),
#                                        color=(255, 0, 0),
#                                        lineType=8))
    
        return [background]
    
    
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

    def get_mean_std_temperature(self, thermal_image, show=False):
        plt.imshow(self.region_mask)
        if show:
            plt.show()
        mean, std = cv2.meanStdDev(thermal_image, mask=self.region_mask)
        return mean.item(), std.item()


class ContourOpenCV:
    SHAPE_TYPES = ['line', 'circle', 'ellipse', 'rectangle', 'poly', 'background']

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
        elif shape_type == 'poly':
            self.__draw_shape = self.__draw_poly
        elif shape_type == 'background':
            self.__draw_shape = self.__draw_background

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

    def __draw_poly(self, image, filled, translation):
        params = self.dict_parameters
        keys = params.keys()

        assert all(key in keys for key in ['pts', 'color'])
        if filled:
            cv2.fillPoly(image, pts=[params['pts']], color=255,
                         lineType=params.get('lineType') or 8,
                         shift=params.get('shift') or 0,
                         offset=params.get('offset') or translation)
        else:
            cv2.polylines(image, pts=[params['pts'] + translation], isClosed=True,
                          color=params['color'],
                          thickness=params.get('thickness') or 1,
                          lineType=params.get('lineType') or 8,
                          shift=params.get('shift') or 0)
            
    def __draw_background(self, image, filled, translation):
        params = self.dict_parameters
        keys = params.keys()
        
        mask = np.zeros(image.shape[:2],np.uint8)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        rect = (20, 20, 900, 1300)
        
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         _, roi = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
#         #cv2.imwrite('/home/dhanushka/stack/roi.png', roi)
#         cont = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         output = np.zeros(image.shape, dtype=np.uint8)
#         cv2.drawContours(output, cont[0], -1, (255, 255, 255))
        
        converted_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(image.shape)
        print(converted_img.shape)
        img=np.uint8(image)
        #image = image.reshape(1440,1080,-1)
        cv2.grabCut(converted_img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        if image.shape == (1440, 1080):
            image = image*mask2
        elif image.shape == (1440, 1080, 3):
            image = image*mask2[:,:,np.newaxis]
        plt.imshow(image),plt.colorbar(),plt.show()
   
        # The final mask is multiplied with  
        # the input image to give the segmented image. 
#        image = img * mask2[:, :, np.newaxis] 

        # output segmented image with colorbar 
        
#        mask[mask > 0] = cv2.GC_PR_FGD
#        mask[mask == 0] = cv2.GC_BGD
#        image = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
