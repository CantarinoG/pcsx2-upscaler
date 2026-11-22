import re
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from support import cutout, surface

import material

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()
ORDER = ("asphalt", "concrete", "dirt", "stone", "masonry", "grass", "wood", "metal", "leather", "fur", "cloth")


def classifier(kind, probabilities):
    session = mock.Mock()
    session.get_inputs.return_value = [mock.Mock(shape=["views", 3, 224, 224])]
    session.run.return_value = [np.array([probabilities], np.float32)]
    with mock.patch.object(material.ort, "InferenceSession", return_value=session):
        return kind(Path("model.onnx"))


class Contract(unittest.TestCase):
    def test_order_is_fixed(self):
        """The models answer by position: families can only be appended."""
        self.assertEqual(tuple(material.FAMILIES), ORDER)

    def test_readme_gives_the_same_order(self):
        self.assertIn("not a surface, " + ", ".join(ORDER) + ".", README)
        self.assertIn(f"`1 x {len(ORDER) + 1}`", README)

    def test_readme_photo_counts(self):
        photos = material.MaterialDetail().photos
        table = dict(re.findall(r"^\| (\w+) \| [^|]+ \| (\d+) \|$", README, re.M))
        self.assertEqual(table, {family: str(len(photos[family])) for family in ORDER})

    def test_each_position_is_its_family(self):
        for i, family in enumerate(ORDER, 1):
            answer = np.eye(len(ORDER) + 1)[i]
            for kind, dump in ((material.MaterialClassifier, surface()), (material.CutoutClassifier, cutout())):
                with self.subTest(family=family, kind=kind.__name__):
                    self.assertEqual(classifier(kind, answer).family(dump), family)

    def test_model_of_the_previous_version_is_rejected(self):
        for kind in (material.MaterialClassifier, material.CutoutClassifier):
            with self.assertRaisesRegex(ValueError, "expected"):
                classifier(kind, [1] + [0] * 8)


class NewFamilies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detail = material.MaterialDetail()

    def change(self, family, up, key="key"):
        out = self.detail.apply(up, (64, 64), family, ("wrap", "wrap"), key)
        plain = up.resize(out.size, Image.Resampling.BICUBIC)
        return np.asarray(out, np.float32) - np.asarray(plain, np.float32)

    def test_strength(self):
        grey = Image.new("RGB", (256, 256), (128, 128, 128))
        got = {family: self.change(family, grey).std() for family in ("leather", "fur", "cloth")}
        for family, expected in (("leather", 0.14), ("fur", 0.14), ("cloth", 0.10)):
            self.assertAlmostEqual(got[family] / (128 * expected), 1, delta=0.35, msg=family)
        self.assertLess(got["cloth"], got["leather"], "cloth is the gentlest")

    def test_fur_runs_the_way_the_texture_does(self):
        across = np.full((256, 256, 3), 120, np.uint8)
        across[(np.arange(256) // 16) % 2 == 0] = 130
        along = across.transpose(1, 0, 2)
        for key in ("a", "b", "c", "d", "e", "f"):
            for image, axis in ((across, 1), (along, 0)):
                change = self.change("fur", Image.fromarray(image), key)[..., 0]
                with self.subTest(key=key, axis=axis):  # strands vary least along their length
                    self.assertLess(np.abs(np.diff(change, axis=axis)).mean(),
                                    np.abs(np.diff(change, axis=1 - axis)).mean())


if __name__ == "__main__":
    unittest.main()
