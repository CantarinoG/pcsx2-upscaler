import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from support import dds_levels, surface

import dds


class Mipmaps(unittest.TestCase):
    def test_colour_survives_low_alpha(self):
        """Roads keep data in alpha: averaging as RGBA turned their smaller levels into static."""
        a = np.zeros((128, 128, 4), np.uint8)
        a[..., :3] = (180, 150, 60)
        a[..., 3] = np.random.default_rng(1).integers(0, 9, (128, 128))
        for level in dds._mip_chain(Image.fromarray(a)):
            np.testing.assert_allclose(np.asarray(level, np.float32)[..., :3].mean(axis=(0, 1)), (180, 150, 60), atol=1)

    def test_cutout_edges_ignore_hidden_colour(self):
        a = np.zeros((64, 64, 4), np.uint8)
        a[..., :3] = (255, 0, 255)  # hidden
        a[:, :21] = (20, 200, 20, 128)
        self.assertTrue(dds.is_cutout(a[..., 3]))
        for level in dds._mip_chain(Image.fromarray(a))[1:]:
            px = np.asarray(level).astype(int)
            shown = px[px[..., 3] > 0][:, :3]
            self.assertLessEqual(np.abs(shown - (20, 200, 20)).max(), 1)

    def test_data_alpha_is_not_a_cutout(self):
        self.assertFalse(dds.is_cutout(np.random.default_rng(1).integers(0, 110, (64, 64))))
        self.assertFalse(dds.is_cutout(np.full((64, 64), 128)))

    def test_file(self):
        with tempfile.TemporaryDirectory() as d:
            for size, saved in (((64, 32), (64, 32)), ((30, 18), (32, 16)), ((1, 1), (4, 4))):
                path = Path(d) / "t.dds"
                dds.save_dds(surface(*size, channels=4), path)
                self.assertEqual(dds_levels(path.read_bytes())[0].shape[:2], saved[::-1])
                self.assertFalse(path.with_name("t.dds.part").exists())


if __name__ == "__main__":
    unittest.main()
