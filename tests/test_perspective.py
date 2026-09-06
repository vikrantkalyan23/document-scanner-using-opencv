import unittest

import numpy as np

from app.scanner.perspective import PerspectiveTransformer


class PerspectiveTransformerTest(unittest.TestCase):
    def test_order_points_returns_clockwise_points(self):
        points = np.array([[100, 200], [10, 10], [110, 15], [20, 190]])

        ordered = PerspectiveTransformer.order_points(points)

        np.testing.assert_array_equal(
            ordered,
            np.array([[10, 10], [110, 15], [100, 200], [20, 190]], dtype="float32"),
        )

    def test_warp_returns_expected_document_shape(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        points = np.array([[20, 10], [119, 10], [119, 89], [20, 89]])

        warped = PerspectiveTransformer.warp(image, points)

        self.assertEqual(warped.shape, (79, 99, 3))


if __name__ == "__main__":
    unittest.main()
