from app.scanner.preprocessing import Preprocessor
from app.scanner.edge_detection import EdgeDetector
from app.scanner.contour_detection import ContourDetector
from app.scanner.perspective import PerspectiveTransformer
from app.scanner.enhancement import ImageEnhancer


class DocumentScanner:
    def process(self, image):
        resized = Preprocessor.resize(image)
        scale = image.shape[1] / resized.shape[1]

        gray = Preprocessor.grayscale(resized)
        blurred = Preprocessor.blur(gray)
        edges = EdgeDetector.canny(blurred)
        document_contour = ContourDetector.find_document_contour(edges)

        if document_contour is None:
            scanned = image.copy()
            original_contour = None
        else:
            original_contour = document_contour.astype("float32") * scale
            scanned = PerspectiveTransformer.warp(image, original_contour)

        enhanced = ImageEnhancer.enhance(scanned)
        contours_image = ContourDetector.draw_document(resized, document_contour)

        print(f"Document detected: {document_contour is not None}")

        return {
            "original": image,
            "resized": resized,
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "contours": contours_image,
            "scanned": scanned,
            "enhanced": enhanced,
            "contour_data": original_contour,
        }
