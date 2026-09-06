from dataclasses import dataclass


@dataclass
class Settings:
    APP_NAME: str = "Document Scanner"
    DEBUG: bool = True

    IMAGE_WIDTH: int = 1200
    MIN_DOCUMENT_AREA_RATIO: float = 0.20
    MAX_CONTOURS: int = 5

    BLUR_KERNEL: tuple = (5, 5)

    CANNY_LOW: int = 75
    CANNY_HIGH: int = 200

    DENOISE_STRENGTH: int = 12
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple = (8, 8)
    SHARPEN_AMOUNT: float = 1.4
    THRESHOLD_BLOCK_SIZE: int = 21
    THRESHOLD_C: int = 10
    NOISE_KERNEL: tuple = (2, 2)


settings = Settings()
