import cv2

from app.config import settings


class ImageEnhancer:
    """
    Improve readability of the scanned document.
    """

    @staticmethod
    def enhance(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = ImageEnhancer.denoise(gray)
        contrast = ImageEnhancer.improve_contrast(denoised)
        sharpened = ImageEnhancer.sharpen(contrast)
        thresholded = ImageEnhancer.threshold(sharpened)

        return ImageEnhancer.remove_specks(thresholded)

    @staticmethod
    def denoise(gray_image):
        return cv2.fastNlMeansDenoising(
            gray_image,
            None,
            settings.DENOISE_STRENGTH,
            7,
            21,
        )

    @staticmethod
    def improve_contrast(gray_image):
        clahe = cv2.createCLAHE(
            clipLimit=settings.CLAHE_CLIP_LIMIT,
            tileGridSize=settings.CLAHE_TILE_GRID_SIZE,
        )

        return clahe.apply(gray_image)

    @staticmethod
    def sharpen(gray_image):
        blurred = cv2.GaussianBlur(gray_image, (0, 0), 1.0)

        return cv2.addWeighted(
            gray_image,
            settings.SHARPEN_AMOUNT,
            blurred,
            -(settings.SHARPEN_AMOUNT - 1.0),
            0,
        )

    @staticmethod
    def threshold(gray_image):
        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            settings.THRESHOLD_BLOCK_SIZE,
            settings.THRESHOLD_C,
        )

    @staticmethod
    def remove_specks(binary_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, settings.NOISE_KERNEL)
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        return cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
        )
