import cv2
import numpy as np

from app.config import settings


class ImageEnhancer:
    """
    Improve readability of the scanned document.
    """

    @staticmethod
    def enhance(image, mode="bw"):
        if mode == "color":
            return ImageEnhancer.enhance_color(image)

        gray = ImageEnhancer.to_gray(image)
        denoised = ImageEnhancer.denoise(gray)
        shadow_free = ImageEnhancer.remove_shadows(denoised)
        contrast = ImageEnhancer.improve_contrast(shadow_free)
        sharpened = ImageEnhancer.sharpen(contrast)

        if mode == "gray":
            return sharpened

        if mode == "soft":
            return ImageEnhancer.soft_threshold(sharpened)

        if mode == "ocr":
            return ImageEnhancer.threshold(
                sharpened,
                settings.OCR_THRESHOLD_BLOCK_SIZE,
                settings.OCR_THRESHOLD_C,
            )

        thresholded = ImageEnhancer.threshold(sharpened)

        return ImageEnhancer.remove_specks(thresholded)

    @staticmethod
    def to_gray(image):
        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def enhance_color(image):
        balanced = ImageEnhancer.white_balance(image)
        denoised = cv2.bilateralFilter(balanced, 7, 35, 35)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        lightness = ImageEnhancer.improve_contrast(lightness)
        enhanced = cv2.merge((lightness, channel_a, channel_b))

        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def white_balance(image):
        result = image.astype("float32")
        averages = result.reshape(-1, 3).mean(axis=0)
        gray_average = averages.mean()
        scale = gray_average / np.maximum(averages, 1.0)
        result *= scale

        return np.clip(result, 0, 255).astype("uint8")

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
    def remove_shadows(gray_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        background = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        background = cv2.medianBlur(background, 21)

        return cv2.divide(gray_image, background, scale=255)

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
    def threshold(gray_image, block_size=None, constant=None):
        block_size = block_size or settings.THRESHOLD_BLOCK_SIZE
        constant = settings.THRESHOLD_C if constant is None else constant
        block_size = ImageEnhancer.ensure_odd_block_size(block_size)

        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )

    @staticmethod
    def soft_threshold(gray_image):
        thresholded = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            ImageEnhancer.ensure_odd_block_size(settings.THRESHOLD_BLOCK_SIZE),
            max(2, settings.THRESHOLD_C // 2),
        )

        return cv2.medianBlur(thresholded, 3)

    @staticmethod
    def remove_specks(binary_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, settings.NOISE_KERNEL)
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        return cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            kernel,
        )

    @staticmethod
    def ensure_odd_block_size(block_size):
        block_size = max(3, int(block_size))

        if block_size % 2 == 0:
            block_size += 1

        return block_size
