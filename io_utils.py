import cv2 as cv
import numpy as np


class PngConst:
    normal_scale = 2 ** 15 - 1
    normal_bias = 1.0
    depth_scale = 2 ** 16 - 1
    depth_bias = 0.0


def write_data_png(data, scale, bias, fn):
    data = ((data + bias) * scale).astype(np.uint16)
    cv.imwrite(fn, data)
    return data


def read_data_png(fn, scale, bias):
    read = cv.imread(fn, cv.IMREAD_UNCHANGED)
    return read.astype(dtype=np.float32) / scale - bias
