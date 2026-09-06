from dataclasses import dataclass


@dataclass
class Settings:
    APP_NAME: str = "Document Scanner"
    DEBUG: bool = True

    IMAGE_WIDTH: int = 1200

    BLUR_KERNEL: tuple = (5, 5)

    CANNY_LOW: int = 75
    CANNY_HIGH: int = 200


settings = Settings()
