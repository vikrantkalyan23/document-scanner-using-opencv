import argparse

import cv2

from app.scanner import DocumentScanner
from app.utils.file_utils import get_input_path, get_output_path, list_input_images
from app.utils.image_utils import print_image_info, save_image, show_image
from app.utils.pdf_utils import save_pdf


SCAN_MODES = ("bw", "color", "gray", "soft", "ocr")


def parse_args():
    parser = argparse.ArgumentParser(description="Scan document image files.")
    parser.add_argument(
        "filename",
        nargs="?",
        default="document.jpeg",
        help="Input filename inside the input directory.",
    )
    parser.add_argument(
        "--mode",
        choices=SCAN_MODES,
        default="bw",
        help="Output enhancement mode.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Scan every supported image in the input directory.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save enhanced scans to a PDF file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the processing windows after saving output files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = list_input_images() if args.batch else [get_input_path(args.filename)]

    if not input_paths:
        print("Error: No supported images found in the input directory.")
        return

    scanner = DocumentScanner()
    pdf_pages = []
    processed_count = 0

    for image_path in input_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        print(f"\nProcessing: {image_path.name}")
        print_image_info(image)

        result = scanner.process(image, mode=args.mode)
        output_paths = save_scan_outputs(result, image_path.stem, args.batch)
        pdf_pages.append(result["enhanced"])
        processed_count += 1

        print("\nGenerated files:")
        for output_path in output_paths:
            print(output_path.name)

        if args.show:
            show_scan_windows(result)

    if args.pdf and pdf_pages:
        pdf_path = get_output_path(pdf_filename(args, processed_count))
        save_pdf(pdf_path, pdf_pages)
        print(f"\nPDF generated: {pdf_path.name}")

    print(f"\nProcessing completed! Images processed: {processed_count}")


def save_scan_outputs(result, stem, use_prefix):
    output_names = [
        ("01_original.jpg", result["original"]),
        ("02_resized.jpg", result["resized"]),
        ("03_grayscale.jpg", result["gray"]),
        ("04_blurred.jpg", result["blurred"]),
        ("05_edges.jpg", result["edges"]),
        ("06_contours.jpg", result["contours"]),
        ("07_scanned.jpg", result["scanned"]),
        ("08_enhanced.jpg", result["enhanced"]),
    ]

    saved_paths = []

    for filename, image in output_names:
        output_name = f"{stem}_{filename}" if use_prefix else filename
        output_path = get_output_path(output_name)
        save_image(output_path, image)
        saved_paths.append(output_path)

    return saved_paths


def show_scan_windows(result):
    show_image("Original", result["original"])
    show_image("Grayscale", result["gray"])
    show_image("Edges", result["edges"])
    show_image("Contours", result["contours"])
    show_image("Enhanced", result["enhanced"])


def pdf_filename(args, processed_count):
    if args.batch:
        return "scanned_documents.pdf"

    if processed_count == 1:
        return f"{args.filename.rsplit('.', 1)[0]}_scan.pdf"

    return "scanned_document.pdf"


if __name__ == "__main__":
    main()
