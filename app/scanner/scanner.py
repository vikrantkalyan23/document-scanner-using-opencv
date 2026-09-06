from app.scanner.preprocessing import Preprocessor
from app.scanner.edge_detection import EdgeDetector
from app.scanner.contour_detection import ContourDetector
from app.scanner.perspective import PerspectiveTransformer
from app.scanner.enhancement import ImageEnhancer
from app.config import settings


class DocumentScanner:
    def process(self, image, mode="bw"):
        detection = self.detect_document(image)
        resized = detection["resized"]
        gray = detection["gray"]
        blurred = detection["blurred"]
        edges = detection["edges"]
        document_contour = detection["document_contour"]

        if detection["original_contour"] is None:
            scanned = image.copy()
            original_contour = None
        else:
            original_contour = detection["original_contour"]
            scanned = PerspectiveTransformer.warp(image, original_contour)

        enhanced = ImageEnhancer.enhance(scanned, mode)
        contours_image = ContourDetector.draw_document(resized, document_contour)

        print(f"Document detected: {original_contour is not None}")

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

    def detect_document(self, image):
        last_result = None

        for width in settings.DETECTION_WIDTHS:
            resized = Preprocessor.resize(image, width)
            scale = image.shape[1] / resized.shape[1]
            gray = Preprocessor.grayscale(resized)
            blurred = Preprocessor.blur(gray)
            edges = EdgeDetector.canny(blurred)
            document_contour = ContourDetector.find_document_contour(edges)

            last_result = {
                "resized": resized,
                "gray": gray,
                "blurred": blurred,
                "edges": edges,
                "document_contour": document_contour,
                "original_contour": None,
            }

            if document_contour is not None:
                last_result["original_contour"] = document_contour.astype("float32") * scale

                return last_result

        return last_result
