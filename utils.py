
import cv2
import numpy as np
from skimage import transform as trans


def reverse_alignment(img, w, h, reverse_align_matrix):
    """
    input :
        img  : numpy array [W, H , 3]
        reverse_align_matrix :
        w : int
        h : int
    outputs:
        reverse_aligned : numpy array [size, size , 3]
    """
    img_for_alignment = img.copy()
    reverse_aligned = cv2.warpAffine(
        img_for_alignment, reverse_align_matrix, (w, h))

    return reverse_aligned


def calculate_reverse_alignment_matrix(m):

    matrix = np.append(m, [[0, 0, 1]], axis=0)
    inverse = np.linalg.inv(matrix)
    inverse = inverse[:2, :]

    return inverse


def calculate_alignment_matrix(src, dst_points):
    """
    input :
        src  : numpy array [5,2]
        size : int  (align image size)
        dst_points :  destination points can be given
    outputs:
        M : numpy array
    """

    tform = trans.estimate_transform("affine", src, dst_points)
    M = tform.params[0:2, :]

    return M


def align_image(img, src, size: int, dst_points):
    """
    input :

        img  : numpy array [W, H , 3]
        src  : numpy array [5,2]
        size : int  (align image size)
        dst_points :  destination points can be given
    outputs:
        aligned : numpy array [size, size , 3]
    """

    img_for_alignment = img.copy()
    tform = trans.estimate_transform("affine", src, dst_points)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img_for_alignment, M, (size, size))

    return aligned.astype("uint8"), M
