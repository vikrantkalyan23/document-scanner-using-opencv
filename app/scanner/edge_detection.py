import cv2

from app.config import settings


class EdgeDetector:
    @staticmethod
    def canny(image):
        """
        Detect edges using Canny edge detection.
        """

        return cv2.Canny(image, settings.CANNY_LOW, settings.CANNY_HIGH)
