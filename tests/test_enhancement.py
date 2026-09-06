import unittest

import numpy as np

from app.scanner.enhancement import ImageEnhancer


class ImageEnhancerTest(unittest.TestCase):
    def test_enhance_returns_binary_grayscale_image(self):
        image = np.full((100, 120, 3), 230, dtype=np.uint8)
        image[30:70, 30:90] = 80

        enhanced = ImageEnhancer.enhance(image)

        self.assertEqual(enhanced.shape, image.shape[:2])
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertTrue(set(np.unique(enhanced)).issubset({0, 255}))

    def test_remove_specks_cleans_isolated_noise(self):
        image = np.full((20, 20), 255, dtype=np.uint8)
        image[10, 10] = 0

        cleaned = ImageEnhancer.remove_specks(image)

        self.assertEqual(cleaned[10, 10], 255)


if __name__ == "__main__":
    unittest.main()
