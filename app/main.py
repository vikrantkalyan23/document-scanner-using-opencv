import argparse

import cv2

from app.scanner import DocumentScanner

from app.utils.image_utils import (
    print_image_info,
    save_image,
    show_image,
)

from app.utils.file_utils import (
    get_input_path,
    get_output_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scan a document image.")
    parser.add_argument(
        "filename",
        nargs="?",
        default="document.jpeg",
        help="Input filename inside the input directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the processing windows after saving output files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------
    # 1. Load input image
    # -----------------------------------

    image_path = get_input_path(args.filename)

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Could not load image:\n{image_path}")
        return

    # -----------------------------------
    # 2. Print image information
    # -----------------------------------

    print_image_info(image)

    # -----------------------------------
    # 3. Create scanner
    # -----------------------------------

    scanner = DocumentScanner()

    # -----------------------------------
    # 4. Process image
    # -----------------------------------

    result = scanner.process(image)

    # -----------------------------------
    # 5. Save processing stages
    # -----------------------------------

    save_image(get_output_path("01_original.jpg"), result["original"])

    save_image(get_output_path("02_resized.jpg"), result["resized"])

    save_image(get_output_path("03_grayscale.jpg"), result["gray"])

    save_image(get_output_path("04_blurred.jpg"), result["blurred"])

    save_image(get_output_path("05_edges.jpg"), result["edges"])

    save_image(get_output_path("06_contours.jpg"), result["contours"])

    save_image(get_output_path("07_scanned.jpg"), result["scanned"])

    save_image(get_output_path("08_enhanced.jpg"), result["enhanced"])

    print("\nProcessing completed!")

    print("\nGenerated files:")

    print("01_original.jpg")
    print("02_resized.jpg")
    print("03_grayscale.jpg")
    print("04_blurred.jpg")
    print("05_edges.jpg")
    print("06_contours.jpg")
    print("07_scanned.jpg")
    print("08_enhanced.jpg")

    # -----------------------------------
    # 6. Display stages
    # -----------------------------------

    if not args.show:
        return

    show_image("Original", result["original"])

    show_image("Grayscale", result["gray"])

    show_image("Edges", result["edges"])

    show_image("Contours", result["contours"])

    show_image("Enhanced", result["enhanced"])


if __name__ == "__main__":
    main()
