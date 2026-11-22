import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from support import cutout, surface

import material

TILING = ("wrap", "wrap")


class Candidates(unittest.TestCase):
    def test_solid_cutout_is_asked_about(self):
        core = material.cutout_core(cutout())
        self.assertIsNotNone(core)
        self.assertEqual(int(core[:, 41:].sum()), 0, "the outline is not part of the core")
        self.assertTrue(core[:, :41].all(), "the texture's own border is not an outline")

    def test_others_are_not(self):
        holes = np.array(surface(64, 64, 4))
        holes[:, 6:, 3] = 0
        soft = np.array(surface(64, 64, 4))
        soft[:, 20:, 3], soft[:, 60:, 3] = 60, 0
        flat = np.array(cutout())
        flat[..., :3] = 90
        cases = {"no alpha": surface(), "opaque": surface(64, 64, 4), "mostly holes": Image.fromarray(holes),
                 "soft": Image.fromarray(soft), "flat": Image.fromarray(flat), "tiny": cutout(24, 24),
                 "huge": cutout(1024, 512), "strip": cutout(256, 32)}
        for name, im in cases.items():
            with self.subTest(name):
                self.assertIsNone(material.cutout_core(im))

    def test_the_ordinary_classifier_still_leaves_cutouts_alone(self):
        self.assertFalse(material.eligible(cutout()))


class Views(unittest.TestCase):
    def test_order_and_sizes(self):
        views = material.cutout_views(cutout(), material.cutout_core(cutout()))
        self.assertTrue(7 <= len(views) <= 9)
        self.assertEqual([v.width for v in views[:6]], [64, 128] * 3)
        self.assertEqual({v.mode for v in views}, {"RGB"})

    def test_hidden_colour_is_never_seen(self):
        plain = cutout()
        loud = np.array(plain)
        loud[loud[..., 3] == 0, :3] = (255, 0, 255)
        loud = Image.fromarray(loud)
        for a, b in zip(material.cutout_views(plain, material.cutout_core(plain)),
                        material.cutout_views(loud, material.cutout_core(loud))):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_patches_come_from_inside(self):
        a = np.array(cutout())
        a[a[..., 3] == 0, :3] = 255
        im = Image.fromarray(a)
        for patch in material.cutout_views(im, material.cutout_core(im))[6:]:
            self.assertLess(np.asarray(patch).max(), 255)


class Classifier(unittest.TestCase):
    def load(self, probabilities):
        self.session = session = mock.Mock()
        session.get_inputs.return_value = [mock.Mock(shape=["views", 3, 224, 224])]
        session.run.return_value = [np.array([probabilities], np.float32)]
        with mock.patch.object(material.ort, "InferenceSession", return_value=session):
            return material.CutoutClassifier(Path("cutout.onnx"))

    def test_probe_and_views(self):
        classifier = self.load([0.2, 0, 0, 0, 0.8, 0, 0, 0, 0])
        self.assertEqual(self.session.run.call_args.args[1][self.session.get_inputs()[0].name].shape[0], 7)
        self.assertEqual(classifier.family(cutout()), "stone")
        self.assertGreaterEqual(self.session.run.call_args.args[1][self.session.get_inputs()[0].name].shape[0], 7)

    def test_not_a_surface(self):
        self.assertIsNone(self.load([0.9, 0, 0, 0, 0.1, 0, 0, 0, 0]).family(cutout()))

    def test_only_cutouts_reach_the_model(self):
        classifier = self.load([0.2, 0, 0, 0, 0.8, 0, 0, 0, 0])
        self.session.run.reset_mock()
        self.assertIsNone(classifier.family(surface()))
        self.session.run.assert_not_called()

    def test_wrong_output_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load([0.5, 0.5])


class Detail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detail = material.MaterialDetail()

    def render(self, dump, masked=True):
        up = dump.resize((256, 256), Image.Resampling.NEAREST)
        mask = material.grain_mask(dump) if masked else None
        return up, np.asarray(self.detail.apply(up, dump.size, "stone", TILING, "key", mask=mask), np.int16)

    def test_nothing_changes_outside_the_solid_part(self):
        up, out = self.render(cutout())
        plain = np.asarray(up.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), np.int16)
        np.testing.assert_array_equal(out[:, 43 * 8:, :3], plain[:, 43 * 8:])
        self.assertGreater(np.abs(out[:, :30 * 8, :3] - plain[:, :30 * 8]).mean(), 2)

    def test_alpha_is_that_of_a_plain_enlargement(self):
        np.testing.assert_array_equal(self.render(cutout())[1][..., 3], self.render(cutout(), False)[1][..., 3])

    def test_inside_is_the_same_grain_as_a_surface(self):
        masked, whole = self.render(cutout())[1], self.render(cutout(), False)[1]
        self.assertLessEqual(np.abs(masked[:, :30 * 8] - whole[:, :30 * 8]).max(), 1)

    def test_slightly_translucent_ends_get_grain_too(self):
        """A band whose end columns have alpha 117 showed a seam where it repeats."""
        a = np.array(surface(64, 64, 4))
        a[:8, :, 3] = 0
        a[8:, (0, -1), 3] = 117
        up, out = self.render(Image.fromarray(a))
        plain = np.asarray(up.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), np.int16)
        change = np.abs(out[200:, :, :3] - plain[200:]).mean(axis=(0, 2))
        self.assertGreater(min(change[:8].mean(), change[-8:].mean()), change.mean() / 2)

    def test_only_cutouts_have_a_mask(self):
        self.assertIsNone(material.grain_mask(surface()))
        self.assertIsNone(material.grain_mask(surface(64, 64, 4, alpha=60)))
        self.assertIsNotNone(material.grain_mask(cutout()))


if __name__ == "__main__":
    unittest.main()
