import cv2
import numpy as np

from app.config import settings


class PerspectiveTransformer:
    """
    Apply a four-point perspective transform to a detected document.
    """

    @staticmethod
    def order_points(points):
        points = np.asarray(points, dtype="float32")

        ordered = np.zeros((4, 2), dtype="float32")
        point_sums = points.sum(axis=1)
        point_diffs = np.diff(points, axis=1).ravel()

        ordered[0] = points[np.argmin(point_sums)]
        ordered[2] = points[np.argmax(point_sums)]
        ordered[1] = points[np.argmin(point_diffs)]
        ordered[3] = points[np.argmax(point_diffs)]

        return ordered

    @staticmethod
    def warp(image, points):
        ordered = PerspectiveTransformer.order_points(points)
        top_left, top_right, bottom_right, bottom_left = ordered

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        max_width = int(round(max(width_top, width_bottom)))

        height_right = np.linalg.norm(top_right - bottom_right)
        height_left = np.linalg.norm(top_left - bottom_left)
        max_height = int(round(max(height_right, height_left)))

        if max_width <= 0 or max_height <= 0:
            return image.copy()

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(ordered, destination)

        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        warped = PerspectiveTransformer.rotate_to_portrait(warped)

        return PerspectiveTransformer.crop_border(warped)

    @staticmethod
    def rotate_to_portrait(image):
        height, width = image.shape[:2]

        if width > height:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image

    @staticmethod
    def crop_border(image):
        height, width = image.shape[:2]
        crop_x = int(width * settings.BORDER_CROP_RATIO)
        crop_y = int(height * settings.BORDER_CROP_RATIO)

        if crop_x == 0 and crop_y == 0:
            return image

        return image[crop_y : height - crop_y, crop_x : width - crop_x]
