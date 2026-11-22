import contextlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from support import dds_levels, surface

import material
import pcsx2_upscale as u


class Gain(unittest.TestCase):
    def test_identical_colour_is_no_gain_whatever_the_alpha(self):
        p = np.random.default_rng(3).integers(0, 40, (64, 64), np.uint8)
        views = []
        for side in (128, 512):
            a = np.full((side, side, 4), (180, 150, 60, 0), np.uint8)
            a[..., 3] = np.asarray(Image.fromarray(p).resize((side, side), 0))
            views.append(Image.fromarray(a))
        self.assertLess(u.second_pass_gain(*views, 12), 1)

    def test_visible_change_is_a_gain(self):
        self.assertGreater(u.second_pass_gain(Image.new("RGB", (128, 128), (90, 90, 90)), surface(512, 512), 16), 3)


class Run(unittest.TestCase):
    """main() on generated dumps, with stand-ins for the models."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = root = Path(tmp.name)
        self.game = root / "textures" / "TEST-00000"
        (self.game / "dumps").mkdir(parents=True)
        (root / "models").mkdir()
        for name in ("text", "other", "material"):
            (root / "models" / f"{name}.onnx").touch()
        self.add("aaaa-1111-00000013", surface(64, 64))         # a surface
        self.add("aaaa-2222-00000013", surface(64, 64, seed=2))  # its palette variant
        self.add("bbbb-1111-00000013", surface(48, 48))         # not a surface
        self.add("cccc-1111-00000013", surface(40, 40))         # text

    def add(self, name, im):
        im.save(self.game / "dumps" / f"{name}.png")

    def size(self, name):
        return u.dds_size(self.game / "replacements" / f"{name}.dds")

    def run_main(self, classifier_error=None):
        classified = self.classified = []

        class Detector:
            def __init__(self, use_cuda):
                pass

            def has_text(self, im):
                return im.width == 40

        class Upscaler:
            provider = "test"

            def __init__(self, path):
                pass

            def upscale_image(self, im):
                big = im.resize((im.width * 4, im.height * 4), Image.Resampling.BICUBIC)
                noise = np.random.default_rng(im.width).integers(-25, 25, (big.height, big.width, 1))
                return Image.fromarray(np.clip(np.asarray(big, np.int16) + noise, 0, 255).astype(np.uint8))

        class Classifier:
            def __init__(self, path):
                if classifier_error:
                    raise classifier_error

            def family(self, im):
                classified.append(im.width)
                return "asphalt" if im.width == 64 else None

        models = self.root / "models"
        patches = dict(TEXT_MODEL=models / "text.onnx", OTHER_MODEL=models / "other.onnx",
                       MATERIAL_MODEL=models / "material.onnx", USAGE_CACHE=self.root / "cache",
                       PLUGINS_DIR=self.root / "plugins", TextDetector=Detector, Upscaler=Upscaler,
                       init_onnxruntime=lambda: None, load_game_titles=lambda textures: {},
                       ProcessPoolExecutor=lambda n, mp_context: ThreadPoolExecutor(2), choose=lambda *a: 0)
        out = io.StringIO()
        with mock.patch.multiple(u, **patches), mock.patch.object(material, "MaterialClassifier", Classifier), \
                mock.patch.object(material.MaterialDetail, "apply", autospec=True,
                                  side_effect=material.MaterialDetail.apply) as apply, \
                mock.patch("sys.argv", ["pcsx2_upscale.py", str(self.game.parent)]), contextlib.redirect_stdout(out):
            u.main()
        self.calls = apply.call_args_list
        return out.getvalue()

    def test_only_surfaces_get_detail(self):
        out = self.run_main()
        self.assertIn("4 written, 0 failed", out)
        self.assertEqual(out.count("(+asphalt)"), 2)
        self.assertNotIn(40, self.classified, "textures with text must not be classified")
        self.assertEqual(self.size("aaaa-1111-00000013"), (512, 512))
        self.assertEqual(self.size("bbbb-1111-00000013"), (192, 192))
        self.assertEqual(self.size("cccc-1111-00000013"), (160, 160))
        for path in (self.game / "replacements").iterdir():
            dds_levels(path.read_bytes())

    def test_palette_variants_pass_the_same_key(self):
        self.run_main()
        self.assertEqual([c.args[-1] for c in self.calls], ["aaaa", "aaaa"])

    def test_second_run_changes_nothing(self):
        self.run_main()
        before = {p.name: p.read_bytes() for p in (self.game / "replacements").iterdir()}
        self.run_main()
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.game / "replacements").iterdir()})

    def test_without_a_classifier_everything_is_plain_4x(self):
        (self.root / "models" / "material.onnx").unlink()
        self.assertNotIn("(+", self.run_main())
        self.assertEqual(self.size("aaaa-1111-00000013"), (256, 256))

    def test_broken_classifier_stops_with_a_message(self):
        with self.assertRaises(SystemExit) as stop:
            self.run_main(classifier_error=ValueError("bad model"))
        self.assertIn("Material detail can't start", str(stop.exception))

    def test_surface_goes_to_16x_and_back(self):
        name = "aaaa-1111-00000013"
        self.run_main()
        usage = self.game / "usage.json"
        usage.write_text(json.dumps({"textures": {name: {"max_mag": 12.0}}}))
        self.assertIn("(+asphalt)", self.run_main())
        self.assertEqual(self.size(name), (1024, 1024))
        usage.write_text(json.dumps({"textures": {name: {"max_mag": 1.0}}}))
        self.assertIn("(+asphalt)", self.run_main())
        self.assertEqual(self.size(name), (512, 512))

    def test_skipped_textures_are_left_unreplaced(self):
        (self.game / "skip.txt").write_text("bbbb-*\n")
        self.run_main()
        self.assertFalse((self.game / "replacements" / "bbbb-1111-00000013.dds").exists())


if __name__ == "__main__":
    unittest.main()
