"""Independent panel experiment: synthetic checks, no camera/robot/model load."""
import ast
import contextlib
import importlib.util
import io
import pickle
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/"scripts/261110_my_panel_generalization.py"
SPEC = importlib.util.spec_from_file_location("panel_experiment", ENTRY)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)
v = experiment.vision


class PanelExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.image = np.full((220, 280, 3), 255, np.uint8)
        cv2.putText(self.image, "3", (125, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 4)
        self.ring = v.Ring(144, 85, 65, 1.)
        self.panel = (60, 20, 230, 170)

    def recognizer(self, raw=None, baseline=None):
        inference = Mock(return_value=raw) if raw is not None else Mock(side_effect=lambda request:
            SimpleNamespace(txts=['3']*len(request.img), scores=[.96]*len(request.img)))
        original = SimpleNamespace(
            min_score=.80, input_type=SimpleNamespace,
            recognize=Mock(return_value=[baseline or v.ButtonObservation(self.ring, status="NO_GLYPH")]),
            engine=SimpleNamespace(text_rec=inference))
        return original, experiment.GeneralizedOCR(original)

    def test_dark_bbox_is_in_full_image_coordinates_without_modifying_input(self):
        before = self.image.copy()
        patches, box = experiment.dark_glyph_variants(self.image, self.ring)
        self.assertEqual(len(patches), 3)
        self.assertEqual(box, (128, 47, 157, 88))
        np.testing.assert_array_equal(self.image, before)

    def test_panel_roi_offset_is_added_once(self):
        original, rec = self.recognizer()
        result = rec.recognize(self.image, [self.ring], self.panel)[0]
        self.assertEqual(result.text_bbox, (128, 47, 157, 88))
        self.assertEqual(result.text, "3")
        self.assertEqual(result.text_center_px, (142, 68))
        self.assertNotEqual(result.text_center_px, result.ring.center)
        self.assertEqual(result.status, "SINGLE_SHOT")
        original.engine.text_rec.assert_called_once()

    def test_clipped_image_boundaries_use_actual_crop_origin(self):
        shifted = cv2.warpAffine(self.image, np.float32([[1, 0, -115], [0, 1, 0]]),
                                (280, 220), borderValue=(255, 255, 255))
        _, box = experiment.dark_glyph_variants(shifted, v.Ring(29, 85, 65, 1.))
        self.assertEqual(box, (13, 47, 42, 88))

    def test_uniform_and_low_contrast_regions_are_not_text(self):
        for level in (0, 128, 255):
            self.assertEqual(experiment.dark_glyph_variants(np.full_like(self.image, level), self.ring), ([], None))
        low = np.full_like(self.image, 128)
        low[50:90, 130:150] = 135
        self.assertEqual(experiment.dark_glyph_variants(low, self.ring), ([], None))

    def test_exact_threshold_low_confidence_conflict_and_symbol_rejection(self):
        self.assertEqual(experiment.dark_vote([("3", .80), ("3", .80)], .80), ("3", .80))
        for variants in ([('3', .79), ('3', .99), ('3', .99)],
                         [('3', .99), ('8', .99), ('3', .99)],
                         [('C', .99), ('1', .99), ('1', .99)],
                         [('HOLD', .99), ('HOLD', .99)], [('3', float('nan')), ('3', .99)]):
            self.assertEqual(experiment.dark_vote(variants, .80), (None, None))

    def test_confirmed_bright_path_is_preserved_without_dark_inference(self):
        old = v.ButtonObservation(self.ring, '4', .98, (120, 40, 150, 90), [('4', .98)]*2)
        original, rec = self.recognizer(baseline=old)
        result = rec.recognize(self.image, [self.ring], self.panel)[0]
        self.assertEqual(result.text, '4')
        self.assertEqual(result.text_bbox, old.text_bbox)
        original.engine.text_rec.assert_not_called()

    def test_conflict_with_bright_numeric_evidence_is_not_hidden(self):
        old = v.ButtonObservation(self.ring, variants=[('8', .95), ('H', .99)])
        _, rec = self.recognizer(baseline=old)
        self.assertIsNone(rec.recognize(self.image, [self.ring], self.panel)[0].text)
        self.assertEqual(rec.diagnostics[0]['dark_rejection'], 'BRIGHT_DARK_CONFLICT')

    def test_malformed_ocr_batch_is_an_explicit_failure(self):
        for raw in (SimpleNamespace(txts=None, scores=None),
                    SimpleNamespace(txts=['3'], scores=[.99]),
                    SimpleNamespace(txts=['3']*3, scores=None)):
            _, rec = self.recognizer(raw)
            with self.assertRaisesRegex(RuntimeError, 'batch structure'):
                rec.recognize(self.image, [self.ring], self.panel)

    def test_no_stale_label_is_carried_to_blank_image(self):
        original, rec = self.recognizer()
        self.assertEqual(rec.recognize(self.image, [self.ring], self.panel)[0].text, '3')
        self.assertIsNone(rec.recognize(np.full_like(self.image, 255), [self.ring], self.panel)[0].text)
        self.assertEqual(rec.diagnostics[0]['dark_variants'], [])
        self.assertEqual(original.engine.text_rec.call_count, 1)

    def test_compact_group_removes_distant_cluster_without_inventing_rings(self):
        main = [v.Ring(x, y, 20, 1.) for x in (300, 350) for y in (100, 150, 200, 250)]
        extra = [v.Ring(50, y, 20, 1.) for y in (100, 150, 200, 250)]
        selected = experiment.compact_group(main+extra)
        self.assertEqual(set(selected), set(main))
        self.assertEqual(experiment.compact_group(main[:3]), [])

    def test_experiment_has_no_camera_or_motion_entry(self):
        tree = ast.parse(ENTRY.read_text())
        names = [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n.startswith(('drivers', 'sdk', 'elevator.app', 'elevator.motion')) for n in names))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in
                             ('move_to_traj', 'make_camera', 'connect', 'set_joint_positions') for n in ast.walk(tree)))

    def test_live_detector_builds_one_engine_and_keeps_new_readings_unstable(self):
        original, _ = self.recognizer()
        original.backend_info = {"actual_device": "cpu"}
        with patch.object(v, 'ButtonOCR', return_value=original) as build:
            detector = v.PanelPerception({}, SimpleNamespace(geometry_only=True,
                ocr_preprocessing='bright_dark', ocr_device='cpu', ocr_min_score=.80))
            for _ in range(2):
                b = detector.ocr.recognize(self.image, [self.ring], self.panel)[0]
                self.assertEqual(b.status, 'UNSTABLE')
                self.assertEqual(b.text, '3')
                self.assertEqual(detector.ocr.last_patch_candidates, 1)
            build.assert_called_once()
        self.assertIs(detector.ocr.original, original)
        self.assertIs(detector.ocr.backend_info, original.backend_info)

    def test_live_dark_result_source_survives_process_serialization(self):
        original, _ = self.recognizer()
        b = v.GeneralizedOCR(original).recognize(self.image, [self.ring], self.panel)[0]
        restored = pickle.loads(pickle.dumps(b))
        self.assertEqual(restored.to_dict()['text_region_source'], 'dark_glyph_components')
        self.assertEqual(restored.text_center_px, (142, 68))
        self.assertEqual(restored.status, 'UNSTABLE')

    def test_live_single_frame_cannot_skip_temporal_confirmation(self):
        from tests.test_temporal import frame
        from elevator.temporal import TemporalButtonTracker
        from dataclasses import replace
        original, _ = self.recognizer()
        rec = v.GeneralizedOCR(original)
        tracker = TemporalButtonTracker(required=3, min_score=.80)
        for at, expected in ((1., 'UNSTABLE'), (2., 'UNSTABLE'), (3., 'STABLE')):
            result = frame(at)
            b = rec.recognize(self.image, [self.ring], self.panel)[0]
            # Place the recognized observation in the known synthetic layout.
            result.buttons[0] = replace(b, ring=v.Ring(100., 100., 18., .97), text_bbox=(95,95,105,105))
            tracker.update(result, now=at)
            self.assertEqual(result.buttons[0].status, expected)

    def test_bright_mode_reuses_original_without_wrapper(self):
        original, _ = self.recognizer()
        original.backend_info = {}
        with patch.object(v, 'ButtonOCR', return_value=original):
            detector = v.PanelPerception({}, SimpleNamespace(geometry_only=True, ocr_preprocessing='bright'))
        self.assertIs(detector.ocr, original)
        with patch.object(v, 'ButtonOCR', side_effect=AssertionError('must validate first')):
            with self.assertRaises(ValueError):
                v.PanelPerception({}, SimpleNamespace(geometry_only=True, ocr_preprocessing='invalid'))

    def test_both_entrypoints_forward_defaults_and_override_before_hardware(self):
        from elevator import app, motion
        source = ROOT/'scripts/261110_my_grasp_then_elevator.py'
        spec = importlib.util.spec_from_file_location('sequence_entry', source)
        sequence = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sequence)
        yaml_path = ROOT/'config/261105_my_elevator.yaml'
        before = yaml_path.read_bytes()
        for main, base in ((app.main, ['--mode','elevator']), (sequence.main, ['--floor','3'])):
            for mode in (None, 'bright', 'bright_dark'):
                argv = base+['--dry-run','--ocr-device','cpu']
                if mode is not None:
                    argv += ['--ocr-preprocessing', mode]
                with self.subTest(entry=main.__module__, mode=mode), \
                        patch.object(app, 'PerceptionWorker', side_effect=RuntimeError('INTERCEPT_BEFORE_DEVICES')) as worker, \
                        patch('drivers.camera.make_camera', side_effect=AssertionError('camera')), \
                        patch.object(motion, 'Robot', side_effect=AssertionError('robot')), \
                        patch('cv2.destroyAllWindows'), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(argv), 2)
                worker.assert_called_once()
                settings = worker.call_args.args[1]
                self.assertEqual(settings['ocr_preprocessing'], mode or 'bright_dark')
                self.assertEqual(settings['ocr_min_score'], app.load_workflow(yaml_path)['ocr']['min_confidence'])
                self.assertEqual(settings['ocr_device'], 'cpu')
        self.assertEqual(yaml_path.read_bytes(), before)

    def test_grasp_mode_does_not_construct_panel_ocr(self):
        from elevator import app, grasp
        with patch.object(grasp, 'run', return_value=0) as run, \
                patch.object(app, 'PerceptionWorker', side_effect=AssertionError('panel OCR')):
            self.assertEqual(app.main(['--mode','grasp','--dry-run','--ocr-preprocessing','bright_dark']), 0)
        run.assert_called_once()

    def test_dark_first_two_views_avoid_unneeded_third_and_do_not_reuse_next_frame(self):
        original, rec = self.recognizer()
        for _ in range(2):
            result = rec.recognize(self.image, [self.ring], self.panel)[0]
            self.assertEqual(result.text, '3')
            self.assertEqual(len(result.variants), 2)
            self.assertEqual(rec.last_inference_stats['inferred_patches'], 2)
        self.assertEqual(original.engine.text_rec.call_count, 2)

    def test_dark_retry_requires_original_support_and_rejects_known_conflict(self):
        for first, binary, expected, calls in [('3', 'H', '3', 2), ('3', '8', None, 1), ('H', '3', None, 1)]:
            original, rec = self.recognizer()
            original.engine.text_rec.side_effect = [
                SimpleNamespace(txts=[first,binary], scores=[.99,.99]),
                SimpleNamespace(txts=['3'], scores=[.98])]
            patches,box=v.dark_glyph_variants(self.image,self.ring)
            patches[1]=patches[1].copy();patches[1][0,0,0]=123
            with patch.object(v,'dark_glyph_variants',return_value=(patches,box)):
                result = rec.recognize(self.image, [self.ring], self.panel)[0]
            self.assertEqual(result.text, expected)
            self.assertEqual(original.engine.text_rec.call_count, calls)

    def test_exact_pixel_reuse_preserves_owner_order_and_recognizes_changed_pixels(self):
        engine = SimpleNamespace(text_rec=Mock(side_effect=lambda request:
            SimpleNamespace(txts=[str(int(p[0,0,0])) for p in request.img], scores=[.95]*len(request.img))))
        frame = v._FrameRecognition(engine, SimpleNamespace)
        a=np.zeros((4,4,3),np.uint8);b=a.copy();b[0,0,0]=3
        self.assertEqual(frame.recognize([a,b,a.copy()]), [('0',.95),('3',.95),('0',.95)])
        self.assertEqual(len(engine.text_rec.call_args.args[0].img), 2)
        self.assertEqual(frame.recognize([b,a]), [('3',.95),('0',.95)])
        engine.text_rec.assert_called_once()
        self.assertEqual(frame.stats['reused_patches'], 3)
        # Same bytes with different shape are distinct inputs, even in one frame.
        frame.recognize([a.reshape(2,8,3)])
        self.assertEqual(engine.text_rec.call_count, 2)

    def test_bright_glyph_is_extracted_once_per_ring_and_blank_skips_ocr(self):
        original, _ = self.recognizer()
        rec=v.ButtonOCR.__new__(v.ButtonOCR)
        rec.engine,rec.input_type,rec.min_score=original.engine,SimpleNamespace,.8
        bright=255-self.image
        with patch.object(v,'glyph_variants',wraps=v.glyph_variants) as extract:
            rec.recognize(bright,[self.ring],self.panel)
        self.assertEqual(extract.call_count,1)
        rec.recognize(np.zeros_like(bright),[self.ring],self.panel)
        self.assertEqual(rec.last_inference_stats['inferred_patches'],0)

    def test_dark_fallback_reuses_original_pixels_from_rejected_bright_branch(self):
        # Force plausible bright/dark glyph boxes with one common original crop.
        base=np.full((20,20,3),30,np.uint8)
        bright_binary=np.full_like(base,80);dark_binary=np.full_like(base,120)
        tight=np.full((10,10,3),180,np.uint8)
        engine=SimpleNamespace(text_rec=Mock(side_effect=lambda request:
            SimpleNamespace(txts=['3' if 30 in p or 120 in p else 'H' for p in request.img],
                            scores=[.95]*len(request.img))))
        rec=v.ButtonOCR.__new__(v.ButtonOCR)
        rec.engine,rec.input_type,rec.min_score=engine,SimpleNamespace,.8
        wrapped=v.GeneralizedOCR(rec)
        with patch.object(v,'glyph_variants',return_value=([base,tight,bright_binary],(1,1,5,5))), \
                patch.object(v,'dark_glyph_variants',return_value=([base,tight,dark_binary],(1,1,5,5))):
            result=wrapped.recognize(self.image,[self.ring],self.panel)[0]
        self.assertEqual(result.text,'3')
        self.assertEqual(wrapped.last_inference_stats['requested_patches'],5)
        self.assertEqual(wrapped.last_inference_stats['inferred_patches'],4)
        self.assertEqual(wrapped.last_inference_stats['reused_patches'],1)


if __name__ == '__main__':
    unittest.main()
