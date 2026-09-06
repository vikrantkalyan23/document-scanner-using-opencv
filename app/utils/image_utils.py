import cv2


def show_image(title: str, image):
    """
    Display an image in a window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(path: str, image):
    """
    Save an image to disk.
    """
    success = cv2.imwrite(str(path), image)

    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def print_image_info(image):
    """
    Print basic image information.
    """
    height, width = image.shape[:2]

    channels = 1 if len(image.shape) == 2 else image.shape[2]

    print("\nImage Information")
    print("-" * 30)
    print(f"Width    : {width}")
    print(f"Height   : {height}")
    print(f"Channels : {channels}")
    print(f"Data Type: {image.dtype}")
