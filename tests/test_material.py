import collections
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from support import surface

import material

TILING = ("wrap", "wrap")


class Photos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detail = material.MaterialDetail()

    def test_every_family_has_several_photos(self):
        for family in material.FAMILIES:
            self.assertGreaterEqual(len(self.detail.photos[family]), 2, family)

    def test_every_folder_is_a_family(self):
        folders = {p.name for p in material.PHOTOS_DIR.iterdir() if p.is_dir()}
        self.assertEqual(folders, set(material.FAMILIES))

    def test_photos_are_credited(self):
        credits = (material.PHOTOS_DIR / "README.md").read_text()
        for family, photos in self.detail.photos.items():
            for name in photos:
                credit = f"`{name}`" if name.startswith("generated_") else f"polyhaven.com/a/{name})"
                self.assertIn(credit, credits)

    def test_photos_are_square_greyscale_and_tile(self):
        for family, photos in self.detail.photos.items():
            for name, photo in photos.items():
                with self.subTest(photo=f"{family}/{name}"):
                    self.assertEqual((photo.size, photo.mode), ((1024, 1024), "L"))
                    a = np.asarray(photo, np.float32)
                    for v in (a, a.T):
                        seam = np.abs(v[0] - v[-1]).mean()
                        self.assertLessEqual(seam, 1.1 * np.abs(np.diff(v, axis=0)).mean(axis=1).max())

    def test_family_without_photos_is_an_error(self):
        with tempfile.TemporaryDirectory() as empty, mock.patch.object(material, "PHOTOS_DIR", Path(empty)):
            with self.assertRaises(FileNotFoundError):
                material.MaterialDetail()


class Choice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detail = material.MaterialDetail()
        cls.keys = [f"{i * 2654435761:016x}" for i in range(1200)]

    def test_key_ignores_palette_and_format(self):
        key = material.texture_key
        self.assertEqual(key("10baad0f55c93c31-1016a0425a74fc8a-000015d4"), key("10baad0f55c93c31-80e52147a23db850-00001dd4"))
        self.assertEqual(key("10baad0f55c93c31-00001dd4"), "10baad0f55c93c31")

    def test_photos_are_used_evenly(self):
        for family, photos in self.detail.photos.items():
            used = collections.Counter(self.detail.choose(family, k) for k in self.keys)
            self.assertEqual(set(used), set(photos), family)
            self.assertGreater(min(used.values()), 0.75 * len(self.keys) / len(photos), family)

    def test_same_choice_in_every_process(self):
        # fixed values: hash() would differ per run
        self.assertEqual(material._seed("abc"), int.from_bytes(bytes.fromhex("a9993e364706816a"), "little"))

    def test_adding_a_photo_only_moves_textures_onto_it(self):
        before = {k: self.detail.choose("stone", k) for k in self.keys}
        more = dict(self.detail.photos["stone"], extra=None)
        with mock.patch.dict(self.detail.photos, stone=more):
            for k in self.keys:
                self.assertIn(self.detail.choose("stone", k), (before[k], "extra"))

    def test_removing_a_photo_only_moves_its_textures(self):
        before = {k: self.detail.choose("stone", k) for k in self.keys}
        gone = sorted(self.detail.photos["stone"])[0]
        fewer = {n: p for n, p in self.detail.photos["stone"].items() if n != gone}
        with mock.patch.dict(self.detail.photos, stone=fewer):
            for k in self.keys:
                if before[k] != gone:
                    self.assertEqual(self.detail.choose("stone", k), before[k])


class Detail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detail = material.MaterialDetail()
        cls.flat = Image.new("RGB", (256, 256), (120, 100, 80))

    def apply(self, family="concrete", key="k", up=None, size=(64, 64), modes=TILING):
        return self.detail.apply(up or self.flat, size, family, modes, key)

    def test_adds_detail_and_keeps_colour(self):
        for family in material.FAMILIES:
            with self.subTest(family=family):
                out = np.asarray(self.apply(family), np.float32)
                self.assertGreater((out @ material.LUMA).std(), 3)
                np.testing.assert_allclose(out.mean(axis=(0, 1)), (120, 100, 80), atol=4)

    def test_contrast_follows_the_family_strength(self):
        for family, strength in material.FAMILIES.items():
            for name in self.detail.photos[family]:
                with self.subTest(photo=f"{family}/{name}"):
                    with mock.patch.object(self.detail, "choose", return_value=name):
                        y = np.asarray(self.apply(family), np.float32) @ material.LUMA
                    self.assertAlmostEqual(float(np.log(y).std()) / strength, 1, delta=0.2)

    def test_same_key_gives_identical_output(self):
        a, b = self.apply(key="same"), material.MaterialDetail().apply(self.flat, (64, 64), "concrete", TILING, "same")
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_different_keys_give_different_output(self):
        outs = {self.apply(key=f"key{i}").tobytes() for i in range(8)}
        self.assertEqual(len(outs), 8)

    def test_offset_differs_even_with_the_same_photo(self):
        name = next(iter(self.detail.photos["stone"]))
        with mock.patch.object(self.detail, "choose", return_value=name):
            self.assertNotEqual(self.apply("stone", "a").tobytes(), self.apply("stone", "b").tobytes())

    def test_palette_variants_get_the_same_grain(self):
        grain = []
        for colour in ((120, 100, 80), (120, 100, 81)):
            up = Image.new("RGB", (256, 256), colour)
            grain.append(np.asarray(self.apply(key="variant", up=up), np.float32) @ material.LUMA)
        self.assertGreater(np.corrcoef(grain[0].ravel(), grain[1].ravel())[0, 1], 0.99)

    def test_shifted_detail_still_tiles(self):
        for family in material.FAMILIES:
            for key in ("a", "b", "c"):
                with self.subTest(family=family, key=key):
                    y = np.asarray(self.apply(family, key), np.float32) @ material.LUMA
                    for v in (y, y.T):
                        seam = np.abs(v[0] - v[-1]).mean()
                        self.assertLessEqual(seam, 1.15 * np.abs(np.diff(v, axis=0)).mean(axis=1).max())

    def test_wood_grain_follows_the_texture(self):
        x = np.arange(256)
        stripes = np.tile((120 + 12 * np.sin(x * 2 * np.pi / 64)).astype(np.uint8), (256, 1))
        for vertical, a in ((True, stripes), (False, stripes.T)):
            base = np.asarray(Image.fromarray(a).convert("RGB").resize((512, 512)), np.float32) @ material.LUMA
            for name in self.detail.photos["wood"]:
                with self.subTest(photo=name, vertical=vertical):
                    with mock.patch.object(self.detail, "choose", return_value=name):
                        out = self.apply("wood", up=Image.fromarray(a).convert("RGB"))
                    grain = np.asarray(out, np.float32) @ material.LUMA - base
                    self.assertEqual(material._runs_vertically(grain, 512), vertical)

    def test_backs_off_on_strong_edges(self):
        a = np.full((256, 256, 3), 60, np.uint8)
        a[:, 128:] = 200
        out = np.asarray(self.apply(up=Image.fromarray(a), modes=("wrap", "symmetric")), np.float32) @ material.LUMA
        change = np.abs(np.log(out + 1) - np.log(np.asarray(Image.fromarray(a).resize((512, 512)), np.float32) @ material.LUMA + 1))
        self.assertLess(change[:, 254:258].mean(), 0.7 * change[:, 100:140].mean())

    def test_alpha_stays_as_it_was(self):
        for value in (0, 60, 128, 255):
            out = self.apply(up=Image.new("RGBA", (256, 256), (120, 100, 80, value)))
            self.assertEqual(out.mode, "RGBA")
            self.assertEqual(set(np.unique(np.asarray(out)[..., 3])), {value})

    def test_colour_is_not_weighted_by_alpha(self):
        up = np.asarray(surface(256, 256, 4)).copy()
        up[..., :3] = (180, 150, 60)
        up[..., 3] = np.random.default_rng(2).integers(0, 9, (256, 256))
        out = np.asarray(self.apply(up=Image.fromarray(up)), np.float32)[..., :3]
        np.testing.assert_allclose(out.mean(axis=(0, 1)), (180, 150, 60), atol=5)

    def test_output_sizes(self):
        for size in ((32, 32), (48, 128), (512, 128), (128, 32), (100, 36), (33, 95), (256, 256)):
            for factor in (4, 16):
                if max(size) * factor > 2048:
                    continue
                with self.subTest(size=size, factor=factor):
                    up = Image.new("RGB", (size[0] * factor, size[1] * factor), (99, 99, 99))
                    scale = max(factor, material.output_scale(*size))
                    self.assertEqual(self.apply(up=up, size=size).size, (size[0] * scale, size[1] * scale))


class Eligibility(unittest.TestCase):
    def test_surface_is_eligible(self):
        self.assertTrue(material.eligible(surface()))
        self.assertTrue(material.eligible(surface(channels=4, alpha=60)))

    def test_left_alone(self):
        cutout = np.asarray(surface(channels=4)).copy()
        cutout[:20, :, 3] = 0
        cases = dict(tiny=surface(16, 16), huge=surface(1024, 256), strip=surface(512, 64),
                     flat=Image.new("RGB", (64, 64), (90, 90, 90)), black=Image.new("RGB", (64, 64)),
                     cutout=Image.fromarray(cutout))
        for name, im in cases.items():
            self.assertFalse(material.eligible(im), name)


class Classifier(unittest.TestCase):
    def make(self, output):
        session = mock.Mock()
        session.get_inputs.return_value = [mock.Mock(shape=["views", 3, 224, 224])]
        session.get_inputs.return_value[0].name = "images"
        session.run.return_value = [np.array([output], np.float32)]
        with mock.patch.object(material.ort, "InferenceSession", return_value=session):
            return material.MaterialClassifier(Path("material.onnx")), session

    def test_each_output_means_its_family(self):
        for i, family in enumerate(material.FAMILIES):
            probs = [0.1] + [0.0] * len(material.FAMILIES)
            probs[i + 1] = 0.9
            self.assertEqual(self.make(probs)[0].family(surface()), family)

    def test_not_a_surface(self):
        probs = [0.51, 0.49] + [0.0] * (len(material.FAMILIES) - 1)
        self.assertIsNone(self.make(probs)[0].family(surface()))

    def test_gets_two_views_in_0_to_1(self):
        classifier, session = self.make([0.1, 0.9] + [0.0] * (len(material.FAMILIES) - 1))
        classifier.family(surface())
        views = next(iter(session.run.call_args.args[1].values()))
        self.assertEqual(views.shape, (2, 3, 224, 224))
        self.assertTrue(0 <= views.min() and views.max() <= 1)

    def test_model_with_other_outputs_is_refused(self):
        for n in (7, 8, 10):
            with self.assertRaises(ValueError):
                self.make([1 / n] * n)


if __name__ == "__main__":
    unittest.main()
