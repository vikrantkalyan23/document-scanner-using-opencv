import unittest

import cv2
import numpy as np

from app.scanner.contour_detection import ContourDetector


class ContourDetectorTest(unittest.TestCase):
    def test_find_document_contour_returns_largest_quad(self):
        edges = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(edges, (40, 30), (260, 270), 255, 2)
        cv2.circle(edges, (25, 25), 8, 255, 1)

        contour = ContourDetector.find_document_contour(edges)

        self.assertIsNotNone(contour)
        self.assertEqual(contour.shape, (4, 2))

    def test_find_document_contour_ignores_small_shapes(self):
        edges = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(edges, (10, 10), (60, 60), 255, 2)

        contour = ContourDetector.find_document_contour(edges)

        self.assertIsNone(contour)


if __name__ == "__main__":
    unittest.main()
