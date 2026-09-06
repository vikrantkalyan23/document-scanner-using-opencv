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

    def test_color_mode_preserves_color_channels(self):
        image = np.full((60, 80, 3), 180, dtype=np.uint8)
        image[:, :, 1] = 210

        enhanced = ImageEnhancer.enhance(image, mode="color")

        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_gray_mode_returns_grayscale_visibility_image(self):
        image = np.full((60, 80, 3), 180, dtype=np.uint8)

        enhanced = ImageEnhancer.enhance(image, mode="gray")

        self.assertEqual(enhanced.shape, image.shape[:2])
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_ensure_odd_block_size_normalizes_values(self):
        self.assertEqual(ImageEnhancer.ensure_odd_block_size(2), 3)
        self.assertEqual(ImageEnhancer.ensure_odd_block_size(10), 11)
        self.assertEqual(ImageEnhancer.ensure_odd_block_size(11), 11)


if __name__ == "__main__":
    unittest.main()
