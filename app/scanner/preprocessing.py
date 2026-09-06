import cv2

from app.config import settings


class Preprocessor:
    @staticmethod
    def resize(image, target_width=None):
        """
        Resize image while maintaining aspect ratio.
        """

        target_width = target_width or settings.IMAGE_WIDTH
        height, width = image.shape[:2]

        if width <= target_width:
            return image

        ratio = target_width / width

        new_width = target_width
        new_height = int(height * ratio)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def grayscale(image):
        """
        Convert BGR image to grayscale.
        """

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def blur(image):
        """
        Apply Gaussian blur to reduce noise.
        """

        return cv2.GaussianBlur(image, settings.BLUR_KERNEL, 0)
