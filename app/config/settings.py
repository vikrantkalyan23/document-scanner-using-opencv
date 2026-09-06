from dataclasses import dataclass


@dataclass
class Settings:
    APP_NAME: str = "Document Scanner"
    DEBUG: bool = True

    IMAGE_WIDTH: int = 1200
    DETECTION_WIDTHS: tuple = (1200, 1000, 800)
    MIN_DOCUMENT_AREA_RATIO: float = 0.20
    MAX_CONTOURS: int = 10
    MIN_DOCUMENT_ASPECT_RATIO: float = 1.0
    MAX_DOCUMENT_ASPECT_RATIO: float = 2.85
    MAX_RIGHT_ANGLE_ERROR: float = 25.0
    BORDER_CROP_RATIO: float = 0.01

    BLUR_KERNEL: tuple = (5, 5)

    CANNY_LOW: int = 75
    CANNY_HIGH: int = 200

    DENOISE_STRENGTH: int = 12
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)
    SHARPEN_AMOUNT: float = 1.4
    THRESHOLD_BLOCK_SIZE: int = 21
    THRESHOLD_C: int = 10
    OCR_THRESHOLD_BLOCK_SIZE: int = 31
    OCR_THRESHOLD_C: int = 15
    NOISE_KERNEL: tuple = (2, 2)
    SUPPORTED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


settings = Settings()
