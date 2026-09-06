import cv2
import numpy as np

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

            if len(approx) == 4 and ContourDetector.is_document_shape(approx):
                return approx.reshape(4, 2)

        return ContourDetector.find_fallback_document_contour(contours, min_area)

    @staticmethod
    def find_fallback_document_contour(contours, min_area):
        for contour in contours[: settings.MAX_CONTOURS]:
            if cv2.contourArea(contour) < min_area:
                continue

            rectangle = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rectangle)

            if ContourDetector.is_document_shape(box):
                return box.astype("float32")

        return None

    @staticmethod
    def is_document_shape(points):
        points = np.asarray(points, dtype="float32").reshape(4, 2)
        width, height = ContourDetector.bounding_dimensions(points)

        if width == 0 or height == 0:
            return False

        aspect_ratio = max(width, height) / min(width, height)

        if not (
            settings.MIN_DOCUMENT_ASPECT_RATIO
            <= aspect_ratio
            <= settings.MAX_DOCUMENT_ASPECT_RATIO
        ):
            return False

        return ContourDetector.max_angle_error(points) <= settings.MAX_RIGHT_ANGLE_ERROR

    @staticmethod
    def bounding_dimensions(points):
        rectangle = cv2.minAreaRect(points)
        width, height = rectangle[1]

        return width, height

    @staticmethod
    def max_angle_error(points):
        ordered = ContourDetector.order_points(points)
        errors = []

        for index in range(4):
            previous_point = ordered[index - 1]
            current_point = ordered[index]
            next_point = ordered[(index + 1) % 4]

            vector_a = previous_point - current_point
            vector_b = next_point - current_point
            denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

            if denominator == 0:
                return 180.0

            cosine = np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine))
            errors.append(abs(90.0 - angle))

        return max(errors)

    @staticmethod
    def order_points(points):
        points = np.asarray(points, dtype="float32").reshape(4, 2)
        ordered = np.zeros((4, 2), dtype="float32")
        point_sums = points.sum(axis=1)
        point_diffs = np.diff(points, axis=1).ravel()

        ordered[0] = points[np.argmin(point_sums)]
        ordered[2] = points[np.argmax(point_sums)]
        ordered[1] = points[np.argmin(point_diffs)]
        ordered[3] = points[np.argmax(point_diffs)]

        return ordered

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
