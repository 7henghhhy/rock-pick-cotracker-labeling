"""CPU checks for the notebook's native-pixel crop and frame/label export path."""
import ast
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch


NOTEBOOK = Path(__file__).resolve().parents[1] / 'rock-pick-cotracker-labeling.ipynb'


def load_helpers() -> dict:
    notebook = json.loads(NOTEBOOK.read_text())
    tree = ast.parse(''.join(notebook['cells'][2]['source']))
    tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    namespace = {'np': np, 'cv2': cv2, 'torch': torch, 'os': os, 'math': math}
    exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
    return namespace


def patterned_frame(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


class CroppingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ns = load_helpers()
        self.points = np.array([[920, 300], [940, 320], [960, 340],
                                [980, 360], [1000, 380]], dtype=float)
        self.visible = np.ones(5, dtype=bool)

    def test_image_center_is_default_and_preserves_pixels(self) -> None:
        self.assertFalse(self.ns['CENTER_ON_KEYPOINTS'])
        frame = patterned_frame(1280, 720)
        original_points = self.points.copy()
        image, points, visible = self.ns['crop_output_frame'](frame, self.points, self.visible)
        np.testing.assert_array_equal(image, frame[120:600, 320:960])
        np.testing.assert_array_equal(points, self.points - [320, 120])
        np.testing.assert_array_equal(visible, [True, True, False, False, False])
        np.testing.assert_array_equal(self.points, original_points)

    def test_optional_keypoint_center_preserves_pixels(self) -> None:
        self.ns['CENTER_ON_KEYPOINTS'] = True
        frame = patterned_frame(1280, 720)
        image, points, visible = self.ns['crop_output_frame'](frame, self.points, self.visible)
        np.testing.assert_array_equal(image, frame[100:580, 640:1280])
        np.testing.assert_array_equal(points, self.points - [640, 100])
        self.assertTrue(visible.all())
        np.testing.assert_array_equal(np.diff(points, axis=0), np.diff(self.points, axis=0))

    def test_small_input_padding_keeps_every_source_pixel(self) -> None:
        frame = patterned_frame(320, 241)
        points = np.array([[20, 20], [40, 40], [60, 60], [80, 80], [100, 100]], dtype=float)
        for mode in (False, True):
            with self.subTest(keypoint_center=mode):
                self.ns['CENTER_ON_KEYPOINTS'] = mode
                image, translated, visible = self.ns['crop_output_frame'](frame, points, self.visible)
                self.assertEqual(image.shape, (480, 640, 3))
                np.testing.assert_array_equal(image[119:360, 160:480], frame)
                self.assertFalse(image[:119].any())
                self.assertFalse(image[360:].any())
                self.assertFalse(image[:, :160].any())
                self.assertFalse(image[:, 480:].any())
                np.testing.assert_array_equal(translated, points + [160, 119])
                self.assertTrue(visible.all())

    def test_resolutions_preserve_keypoint_distances_in_both_modes(self) -> None:
        resolutions = [(1920, 1080), (1280, 720), (640, 480), (320, 240),
                       (1080, 1920), (480, 640), (641, 479)]
        for width, height in resolutions:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for mode in (False, True):
                for span in (20, 80):
                    with self.subTest(resolution=(width, height), mode=mode, span=span):
                        self.ns['CENTER_ON_KEYPOINTS'] = mode
                        points = np.array([[width // 2 + dx, height // 2 + dy]
                                           for dx, dy in [(-span, -span), (-10, -10),
                                                          (0, 0), (10, 10), (span, span)]], dtype=float)
                        image, translated, visible = self.ns['crop_output_frame'](frame, points, self.visible)
                        self.assertEqual(image.shape, (480, 640, 3))
                        np.testing.assert_array_equal(np.diff(translated, axis=0), np.diff(points, axis=0))
                        self.assertTrue(visible.all())

    def test_keypoint_center_ignores_hidden_points_and_clamps_at_edge(self) -> None:
        self.ns['CENTER_ON_KEYPOINTS'] = True
        frame = patterned_frame(1280, 720)
        points = np.array([[10, 10], [20, 20], [30, 30], [np.nan, 50], [1270, 700]])
        image, translated, visible = self.ns['crop_output_frame'](
            frame, points, np.array([True, True, True, True, False]))
        np.testing.assert_array_equal(image, frame[:480, :640])
        np.testing.assert_array_equal(translated[:3], points[:3])
        np.testing.assert_array_equal(visible, [True, True, True, False, False])

    def test_no_visible_points_falls_back_to_image_center(self) -> None:
        self.ns['CENTER_ON_KEYPOINTS'] = True
        frame = patterned_frame(1280, 720)
        image, _, visible = self.ns['crop_output_frame'](frame, self.points, np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(image, frame[120:600, 320:960])
        self.assertFalse(visible.any())

    def test_empty_crop_previews_return_without_plotting(self) -> None:
        # Center cropping can reject every frame when the pick is near an edge.
        self.assertIsNone(self.ns['show_previews']('camera', []))

    def test_label_excludes_points_outside_crop(self) -> None:
        points = np.array([[10, 20], [100, 120], [640, 30], [np.nan, 0], [-5, 30]])
        values = np.array(self.ns['yolo_line'](points, self.visible, 640, 480, 0).split(), dtype=float)
        self.assertEqual(len(values), 20)
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values[1:5], [55 / 640, 70 / 480, 90 / 640, 100 / 480], atol=1e-6)
        np.testing.assert_array_equal(values[7::3], [2, 2, 0, 0, 0])
        np.testing.assert_array_equal(values[11:], np.zeros(9))

    def test_actual_frame_and_label_export_with_cpu_tracker_stub(self) -> None:
        """Exercise normal export using real torch tensors and a stationary-point stub."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video_path = root / 'camera.avi'
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (856, 482))
            if not writer.isOpened():
                self.skipTest('OpenCV MJPG encoder is unavailable')
            frame = np.full((482, 856, 3), 90, dtype=np.uint8)
            for _ in range(5):
                writer.write(frame)
            writer.release()
            points = np.array([[210, 100], [300, 150], [400, 200], [500, 250], [780, 300]], dtype=float)
            calls = []

            def tracker(video, queries, backward_tracking):
                calls.append((tuple(video.shape), queries.clone()))
                tracks = queries[:, :, 1:].unsqueeze(1).expand(1, video.shape[1], 5, 2).clone()
                return tracks, torch.ones((1, video.shape[1], 5))

            self.ns.update({'device': 'cpu', 'model': tracker, 'EVERY': 2, 'TRACK_STRIDE': 2})
            for mode, offset in [(False, [108, 1]), (True, [175, 0])]:
                with self.subTest(keypoint_center=mode):
                    self.ns['CENTER_ON_KEYPOINTS'] = mode
                    self.ns['OUT'] = str(root / str(mode))
                    clip, previews = self.ns['track_and_save'](str(video_path), points.tolist(), 0)
                    self.assertEqual(clip, 'camera')
                    self.assertEqual(len(previews), 3)
                    images = sorted((Path(self.ns['OUT']) / 'train/images').glob('*.jpg'))
                    labels = sorted((Path(self.ns['OUT']) / 'train/labels').glob('*.txt'))
                    self.assertEqual(len(images), 3)
                    self.assertEqual(len(labels), 3)
                    for image_path, label_path in zip(images, labels):
                        self.assertEqual(cv2.imread(str(image_path)).shape, (480, 640, 3))
                        values = np.array(label_path.read_text().split(), dtype=float)
                        self.assertEqual(len(values), 20)
                        expected = (points - offset) / [640, 480]
                        count = 5 if mode else 4
                        np.testing.assert_allclose(values[5::3][:count], expected[:count, 0], atol=1e-6)
                        np.testing.assert_allclose(values[6::3][:count], expected[:count, 1], atol=1e-6)
                        np.testing.assert_array_equal(values[7::3], [2, 2, 2, 2, 2 if mode else 0])
                        if not mode:
                            np.testing.assert_array_equal(values[-3:], [0, 0, 0])
            self.assertEqual(calls[0][0], (1, 3, 3, 360, 640))
            np.testing.assert_allclose(calls[0][1][0, :, 1:].numpy(), points * [640 / 856, 360 / 482], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
