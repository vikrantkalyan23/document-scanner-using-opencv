import cv2


class ContourDetector:
    @staticmethod
    def find(edges):
        """
        Find contours in an edge image.
        """

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    @staticmethod
    def draw(image, contours):
        """
        Draw all detected contours on an image.
        """

        result = image.copy()

        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        return result
