import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.utils.pdf_utils import save_pdf


class PdfUtilsTest(unittest.TestCase):
    def test_save_pdf_writes_file(self):
        image = np.full((20, 30), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"

            save_pdf(path, [image])

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
