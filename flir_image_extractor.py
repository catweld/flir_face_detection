#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code based on https://github.com/Nervengift/read_thermal.py/blob/master/flir_image_extractor.py

import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
from PIL import Image
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2.cv2 as cv2

import numpy as np

from skimage.transform import resize


class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.jpg"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None
        self.img_thermal_rgb = None

        # These values refer to the rgb image
        self.height = None
        self.width = None

    def process_image(self, flir_img_filename, upsample_thermal=False, transform_rgb=False):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)

        Parameters:
            flir_img_filename: path of the rgb+thermal image.
            upsample_thermal: resize the thermal image
        :return:
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        self.rgb_image_np = self.extract_embedded_image()
        self.height, self.width = self.rgb_image_np.shape[:2]
        self.thermal_image_np = self.extract_thermal_image()

        if upsample_thermal:
            self.thermal_image_np = resize(self.thermal_image_np, self.rgb_image_np.shape[:2])

        thermal_np = self.thermal_image_np
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        self.img_thermal_rgb = np.uint8(cm.inferno(thermal_normalized)[:, :, :3] * 255)

        # Transform rgb image
        # From https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
        # cv2.imshow("RGB image before rescale", self.rgb_image_np)
        # cv2.waitKey(0)

        if transform_rgb:
            scale = 0.81
            x_shift = 0
            y_shift = 37
            new_height = round(self.height * scale)
            new_width = round(self.width * scale)
            x_start = (self.width - new_width) // 2
            x_end = self.width - x_start
            y_start = (self.height - new_height) // 2
            y_end = self.height - y_start
            pts1 = np.float32([[x_shift + x_start, y_shift + y_start], [x_shift + x_end, y_shift + y_start],
                               [x_shift + x_start, y_shift + y_end], [x_shift + x_end, y_shift + y_end]])
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            self.rgb_image_np = cv2.warpPerspective(self.rgb_image_np, M, (self.width, self.height))

            # Attempt rotation for horizontal images
            (h, w) = self.img_thermal_rgb.shape[:2]
            if h < w:
                center = (w / 2, h / 2)
                angle90 = 90
                scale = 1.0
                M = cv2.getRotationMatrix2D(center, angle90, scale)
                self.img_thermal_rgb = cv2.warpAffine(self.img_thermal_rgb, M, (h, w))

            (h, w) = self.rgb_image_np.shape[:2]
            if h < w:
                center = (w / 2, h / 2)
                angle90 = 90
                scale = 1.0
                M = cv2.getRotationMatrix2D(center, angle90, scale)
                self.rgb_image_np = cv2.warpAffine(self.rgb_image_np, M, (h, w))

            (h, w) = self.thermal_image_np.shape[:2]
            if h < w:
                center = (w / 2, h / 2)
                angle90 = 90
                scale = 1.0
                M = cv2.getRotationMatrix2D(center, angle90, scale)
                self.thermal_image_np = cv2.warpAffine(self.thermal_image_np, M, (h, w))

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]

        return meta['RawThermalImageType']

    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j'])
        meta = json.loads(meta_json.decode())[0]

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output(
            [self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = self.default_distance
        if 'SubjectDistance' in meta:
            subject_distance = FlirImageExtractor.extract_float(meta['SubjectDistance'])

        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x, E=meta['Emissivity'], OD=subject_distance,
                                                                          RTemp=FlirImageExtractor.extract_float(
                                                                              meta['ReflectedApparentTemperature']),
                                                                          ATemp=FlirImageExtractor.extract_float(
                                                                              meta['AtmosphericTemperature']),
                                                                          IRWTemp=FlirImageExtractor.extract_float(
                                                                              meta['IRWindowTemperature']),
                                                                          IRT=meta['IRWindowTransmission'],
                                                                          RH=FlirImageExtractor.extract_float(
                                                                              meta['RelativeHumidity']),
                                                                          PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                          PF=meta['PlanckF'],
                                                                          PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                 PR2=0.012545258):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        # temperature from radiance
        temp_celcius = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        """
        Extract the float value of a string, helpful for parsing the exiftool data
        :return:
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def plot(self):
        """
        Plot the rgb + thermal image (easy to see the pixel values)
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()

        plt.subplot(1, 2, 1)
        plt.imshow(thermal_np, cmap='hot')
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_np)
        plt.show()

    def save_images(self):
        """
        Save the extracted images
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.extract_thermal_image() if self.thermal_image_np is None else self.thermal_image_np

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized)[:, :, :3] * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        image_filename = fn_prefix + self.image_suffix
        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))

        img_visual.save(image_filename)
        img_thermal.save(thermal_filename)

    def export_thermal_to_csv(self, csv_filename):
        """
        Convert thermal data in numpy to json
        :return:
        """

        with open(csv_filename, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'temp (c)'])

            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])

            writer.writerows(pixel_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=False)
    parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
                        required=False)
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)
    fie.process_image(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()
