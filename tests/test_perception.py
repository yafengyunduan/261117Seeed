"""Offline checks for the current implementation."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from elevator import perception as vision

class FakeOCR:
    backend_info = {"actual_device": "MOCK", "recognition_only": True}

    def __init__(self, *args, **kwargs):
        pass

    def recognize(self, image, rings, box):
        return [vision.ButtonObservation(ring) for ring in rings]


def perception(**overrides):
    args = SimpleNamespace(geometry_only=True, **overrides)
    with patch.object(vision, "ButtonOCR", FakeOCR):
        return vision.PanelPerception({}, args)


class DetectionChecks(unittest.TestCase):
    def test_yolo_warmup_creates_no_observations_or_tracking_history(self):
        p=perception();p.model=object()
        with patch.object(vision,"detect_objects",return_value=([],[])) as infer:
            p.warmup(480,848)
        self.assertEqual(infer.call_args.args[1].shape,(480,848,3))
        self.assertEqual(p.calls,0);self.assertEqual(p.hough_calls,0)
        self.assertIsNone(p.layout.previous)

    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(4)
        cls.paths = [ROOT/"readme/mmexport1793686157333.jpg", ROOT/"readme/mmexport1793686160331.jpg"]
        cls.photos = [cv2.imread(str(path)) for path in cls.paths]
        if any(image is None for image in cls.photos):
            raise RuntimeError("The two original photos are required for the layout tests")

    def test_source_switch_does_not_change_roi(self):
        p = perception()
        p.model = object()
        image = self.photos[0]
        detection = SimpleNamespace(class_name="elevator control panel", conf=.9,
                                    bbox_xyxy=(0, 0, image.shape[1], image.shape[0]))
        with patch.object(vision, "detect_objects", return_value=([], [detection])):
            first = p.process(image, captured_at=1.)
            second = p.process(image, captured_at=2.)
        self.assertEqual(first.panel_source, "YOLOE")
        self.assertEqual(second.panel_source, "GEOMETRY_PROPOSAL")
        np.testing.assert_allclose(second.panel_bbox, first.panel_bbox, atol=2)
        self.assertEqual(second.tracking["geometry_mode"], "TRACKED_CURRENT_EDGES")
        self.assertEqual(second.yolo_detections, [])  # No cached YOLO evidence.

    def test_translation_brightness_and_blur_keep_both_layouts(self):
        for image in self.photos:
            with self.subTest(shape=image.shape):
                p = perception()
                baseline = p.process(image, captured_at=1.)
                self.assertEqual(len(baseline.buttons), 12)
                for index, (dx, dy, gain, blur) in enumerate(((2, 1, 1., False), (4, 2, .87, False), (6, 3, 1., True)), 2):
                    frame = cv2.warpAffine(image, np.array([[1., 0., dx], [0., 1., dy]]),
                                           (image.shape[1], image.shape[0]))
                    frame = cv2.convertScaleAbs(frame, alpha=gain)
                    if blur:
                        frame = cv2.GaussianBlur(frame, (3, 3), .6)
                    result = p.process(frame, captured_at=float(index))
                    self.assertEqual(len(result.buttons), 12)
                    self.assertEqual(result.tracking["geometry_mode"], "TRACKED_CURRENT_EDGES")
                    expected = np.array(baseline.panel_bbox) + np.array([dx, dy, dx, dy])
                    np.testing.assert_allclose(result.panel_bbox, expected, atol=3)
                self.assertEqual(p.hough_calls, 1)

    def test_one_occluded_rim_does_not_shrink_panel_or_invent_button(self):
        image = self.photos[0]
        p = perception()
        first = p.process(image, captured_at=1.)
        ring = min((button.ring for button in first.buttons), key=lambda r: r.y)
        covered = image.copy()
        margin = int(ring.radius*1.25)
        x, y = ring.center
        cv2.rectangle(covered, (x-margin, y-margin), (x+margin, y+margin), (45, 45, 45), -1)
        result = p.process(covered, captured_at=2.)
        self.assertIsNotNone(result.panel_bbox)
        self.assertLess(len(result.buttons), len(first.buttons))
        np.testing.assert_allclose(result.panel_bbox, first.panel_bbox, atol=3)
        self.assertTrue(all(np.hypot(b.ring.x-ring.x, b.ring.y-ring.y) > ring.radius for b in result.buttons))

    def test_blank_after_detection_returns_no_panel_and_clears_layout(self):
        p = perception()
        image = self.photos[0]
        p.process(image, captured_at=1.)
        result = p.process(np.zeros_like(image), captured_at=2.)
        self.assertIsNone(result.panel_bbox)
        self.assertEqual(result.buttons, [])
        self.assertIsNone(p.layout.previous)

    def test_current_yolo_without_current_geometry_is_not_panel(self):
        p = perception()
        p.model = object()
        image = np.zeros((480, 640, 3), np.uint8)
        detection = SimpleNamespace(class_name="elevator control panel", conf=.99, bbox_xyxy=(10, 10, 600, 450))
        with patch.object(vision, "detect_objects", return_value=([], [detection])):
            result = p.process(image, captured_at=1.)
        self.assertIsNone(result.panel_bbox)
        self.assertEqual(result.buttons, [])
        self.assertEqual(len(result.yolo_detections), 1)

    def test_periodic_hough_miss_preserves_current_verified_tracking(self):
        p = perception(hough_refresh_every=2)
        image = self.photos[1]
        first = p.process(image, captured_at=1.)
        with patch.object(vision, "find_rings", return_value=[]):
            second = p.process(image, captured_at=2.)
        self.assertTrue(second.tracking["hough_this_frame"])
        self.assertEqual(second.tracking["geometry_mode"], "TRACKED_CURRENT_EDGES")
        self.assertEqual(len(second.buttons), 12)
        np.testing.assert_allclose(second.panel_bbox, first.panel_bbox, atol=2)

    def test_current_ocr_double_has_no_inferred_or_cached_numbers(self):
        p = perception()
        first = p.process(self.photos[0], captured_at=1.)
        first.buttons[0].text = "3"
        second = p.process(self.photos[0], captured_at=2.)
        self.assertTrue(all(button.text is None for button in second.buttons))
        self.assertEqual(second.to_dict()["labels_source"], "CURRENT_OCR_ONLY")

    def test_result_and_legacy_ring_pickle_under_shared_module_names(self):
        result = vision.FrameResult(1., (100, 100), buttons=[vision.ButtonObservation(vision.Ring(20., 30., 8., .9))])
        restored = pickle.loads(pickle.dumps(result))
        self.assertEqual(restored.buttons[0].ring.center, (20, 30))
        self.assertIs(type(restored), vision.FrameResult)
        self.assertIs(vision.Ring, sys.modules["elevator.perception"].Ring)

    def test_recognition_backend_is_built_once_and_receives_model_directory(self):
        build = Mock(return_value=(object(), {"actual_device": "MOCK"}))
        backend = SimpleNamespace(build_ocr=build)
        model_dir = ROOT/"models/261104_my_rapidocr_torch"
        with patch.object(vision, "backend", backend):
            ocr = vision.ButtonOCR(4, .85, "cuda", 0, model_dir)
        build.assert_called_once_with(threads=4, device="cuda", gpu_index=0, model_dir=model_dir)
        self.assertEqual(ocr.backend_info["actual_device"], "MOCK")



class SharedPanelWorkflowChecks(unittest.TestCase):
    def setUp(self):
        from elevator import panel_261112
        self.shared=panel_261112

    def test_shared_result_pickle_has_importable_package_type(self):
        result=self.shared.EvidenceFrameResult(1.,(100,100),buttons=[vision.ButtonObservation(vision.Ring(20,30,8,1.))],
            tracking={'workflow_evidence':True,'display_label_cache':{}})
        restored=pickle.loads(pickle.dumps(result))
        self.assertEqual(type(restored).__module__,'elevator.panel_261112')
        self.assertEqual(restored.buttons[0].ring.center,(20,30))
        self.assertEqual(restored.to_dict()['labels_source'],'CURRENT_OCR_WITH_SEPARATE_PIXEL_MATCHED_DISPLAY_CACHE')

    def test_workflow_strips_cached_text_but_display_does_not_change_motion_result(self):
        import copy
        args=SimpleNamespace(geometry_only=True,workflow_evidence=True,ocr_buttons_per_update=2)
        engine=Mock(backend_info={},last_inference_stats={})
        engine.recognize.side_effect=lambda im,rings,box:[vision.ButtonObservation(r,str(i+1),.95,
                                    (int(r.x-4),int(r.y-10),int(r.x+4),int(r.y+10))) for i,r in enumerate(rings)]
        with patch.object(vision,'ButtonOCR',return_value=engine):detector=self.shared.FastPanelPerception({},args)
        # A textured native frame gives reliable registration without hardware.
        im=np.full((500,500,3),70,np.uint8)
        rings=[vision.Ring(x,y,32,1.) for y in (150,270) for x in (180,300)]
        for r in rings:
            cv2.circle(im,r.center,32,(230,230,230),3)
            cv2.putText(im,'4',(int(r.x-10),int(r.y+8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        with patch.object(vision,'find_rings',return_value=rings),patch.object(vision,'repeated_button_group',side_effect=lambda x:x), \
                patch.object(self.shared,'recover_panel_rims',return_value=([],{})):
            for k in range(8):last=detector.process(im,captured_at=10+k*.2)
        self.assertTrue(last.tracking['display_label_cache'])
        for key in last.tracking['display_label_cache']:
            self.assertIsNone(last.buttons[int(key)].text)
        before=copy.deepcopy(last.to_dict())
        display=self.shared.draw_workflow_result(im,last)
        self.assertEqual(display.shape,im.shape)
        self.assertEqual(last.to_dict(),before)
        self.assertEqual(engine.recognize.call_count,6)
