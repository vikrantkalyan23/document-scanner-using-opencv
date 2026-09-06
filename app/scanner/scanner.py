from app.scanner.preprocessing import Preprocessor
from app.scanner.edge_detection import EdgeDetector
from app.scanner.contour_detection import ContourDetector


class DocumentScanner:
    def process(self, image):

        # Step 1: Resize
        resized = Preprocessor.resize(image)

        # Step 2: Grayscale
        gray = Preprocessor.grayscale(resized)

        # Step 3: Blur
        blurred = Preprocessor.blur(gray)

        # Step 4: Edge Detection
        edges = EdgeDetector.canny(blurred)

        # Step 5: Find Contours
        contours = ContourDetector.find(edges)

        # Step 6: Draw contours
        contours_image = ContourDetector.draw(resized, contours)

        print(f"Contours Found : {len(contours)}")

        return {
            "original": image,
            "resized": resized,
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "contours": contours_image,
            "contour_data": contours,
        }
