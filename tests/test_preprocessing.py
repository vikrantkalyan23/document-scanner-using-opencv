import unittest

import numpy as np

from app.config import settings
from app.scanner.preprocessing import Preprocessor


class PreprocessorTest(unittest.TestCase):
    def test_resize_downscales_wide_images_only(self):
        image = np.zeros((1000, 2400, 3), dtype=np.uint8)

        resized = Preprocessor.resize(image)

        self.assertEqual(resized.shape[:2], (500, settings.IMAGE_WIDTH))

    def test_resize_keeps_small_images_unchanged(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        resized = Preprocessor.resize(image)

        self.assertIs(resized, image)


if __name__ == "__main__":
    unittest.main()
