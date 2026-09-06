import cv2


def save_pdf(path, images):
    """
    Save OpenCV images as a multi-page PDF.
    """

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires Pillow. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    pages = [to_pillow_image(image) for image in images]

    if not pages:
        raise ValueError("No images were provided for PDF export.")

    first_page, remaining_pages = pages[0], pages[1:]
    first_page.save(
        path,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=remaining_pages,
    )


def to_pillow_image(image):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires Pillow. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    if len(image.shape) == 2:
        return Image.fromarray(image).convert("RGB")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb_image).convert("RGB")
