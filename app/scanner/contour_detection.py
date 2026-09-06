import cv2

from app.config import settings


class ContourDetector:
    @staticmethod
    def find(edges):
        """
        Find contours in an edge image.
        """

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return contours

    @staticmethod
    def find_document_contour(edges):
        """
        Return the largest four-sided contour that is likely to be a document.
        """

        image_area = edges.shape[0] * edges.shape[1]
        min_area = image_area * settings.MIN_DOCUMENT_AREA_RATIO
        contours = ContourDetector.find(edges)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[: settings.MAX_CONTOURS]:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    @staticmethod
    def draw(image, contours):
        """
        Draw all detected contours on an image.
        """

        result = image.copy()

        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result

    @staticmethod
    def draw_document(image, contour):
        """
        Draw the detected document contour on an image.
        """

        result = image.copy()

        if contour is not None:
            cv2.drawContours(result, [contour.astype("int32")], -1, (0, 255, 0), 3)

        return result
