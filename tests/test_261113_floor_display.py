"""Offline checks; model, camera and UI are mocked except geometry on photos."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from elevator import floor_display_261113 as floor

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'scripts/261113_my_floor_display.py'
spec = importlib.util.spec_from_file_location('floor_preview', ENTRY)
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)


def fixture(text='6'):
    image = np.full((500, 400, 3), 170, np.uint8)
    cv2.rectangle(image, (65, 25), (335, 420), (12, 12, 12), -1)
    cv2.rectangle(image, (105, 140), (295, 220), (65, 48, 48), -1)
    if text:
        cv2.putText(image, text, (225, 201), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), 3, cv2.LINE_AA)
    return image, np.float32([[105, 140], [295, 140], [295, 220], [105, 220]])


def fake_ocr(value='6', confidence=.98):
    return NS(info={'actual_backend':'FAKE_TEST', 'actual_device':'test'}, read=Mock(return_value=((value, confidence), 1.5)))


class GeometryTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)

    def test_four_photos_locate_lcd_without_model_or_filename_floor_hints(self):
        expected = {'3':(290, 135, 425, 195), '5':(330, 202, 450, 280),
                    '6':(343, 218, 460, 282), '7':(334, 233, 460, 305)}
        with patch.object(floor, 'build_ocr', side_effect=AssertionError('no model for localization')):
            for label, box in expected.items():
                image = cv2.imread(str(ROOT/f'readme/PairDrop_files_20261112_1539/电梯楼层-{label}.jpg'))
                quad, reason = floor.find_display(image)
                self.assertEqual(reason, 'DISPLAY_FOUND', label)
                screen, inverse, text, metrics = floor._screen_check(image, quad)
                self.assertIsNotNone(text)
                x, y = np.mean(quad, axis=0)/4.096
                self.assertTrue(box[0] < x < box[2] and box[1] < y < box[3])
                self.assertLess(np.prod(text[0].shape[:2]), .01*np.prod(image.shape[:2]))
                self.assertGreaterEqual(metrics['dark_surround_fraction'], .70)

    def test_blank_colored_nameplate_and_circle_buttons_are_not_displays(self):
        blank = np.full((500, 400, 3), 70, np.uint8)
        self.assertIsNone(floor.find_display(blank)[0])
        image, quad = fixture()
        red = image.copy(); cv2.fillConvexPoly(red, quad.astype(np.int32), (50, 45, 150))
        self.assertIsNone(floor._screen_check(red, quad))
        self.assertIsNone(floor._screen_check(image, quad+np.float32([500, 0])))
        # The lower button panel contains numbers but no cabin display.
        image = cv2.imread(str(ROOT/'readme/mmexport1793686157333.jpg'))
        self.assertIsNone(floor.find_display(image)[0])

    def test_inverse_perspective_maps_screen_text_back_to_original_pixels(self):
        image = np.full((500, 600, 3), 40, np.uint8)
        quad = np.float32([[100, 100], [400, 120], [420, 230], [110, 220]])
        screen, inverse = floor.rectify(image, quad[[2, 0, 3, 1]])
        h, w = screen.shape[:2]
        corners = np.float32([[[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]])
        actual = cv2.perspectiveTransform(corners, inverse)[0]
        np.testing.assert_allclose(actual, quad, atol=1e-3)
        with self.assertRaises(ValueError): floor.order_quad([[0, 0]]*4)
        with self.assertRaises(ValueError): floor.order_quad([[0, 0], [1, 0], [1, float('nan')], [0, 1]])

    def test_layout_crop_excludes_left_arrow_and_accepts_multi_digit_text(self):
        image, quad = fixture()
        cv2.putText(image, 'V', (130, 201), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), 3)
        screen, inverse = floor.rectify(image, quad)
        patch, box, signature = floor.text_region(screen)
        self.assertGreater(box[0], .5*screen.shape[1])
        multi = np.full((90, 260, 3), 45, np.uint8)
        cv2.putText(multi, '12', (145, 63), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (220, 220, 220), 3)
        self.assertIsNotNone(floor.text_region(multi))
        self.assertIsNone(floor.text_region(np.full((90, 260, 3), 40, np.uint8)))

    def test_normalization_never_turns_arrows_or_letters_into_digits(self):
        for source, target in [('３','3'), ('-2','-2'), ('B12','B12'), (' 7 ','7'), ('01','1'), ('−1','-1')]:
            self.assertEqual(floor.normalize_floor(source), target)
        for source in ('V5', 'v', 'S', 'A', 'E-4', '1350KG', '18人', '3.0', '3 4', '', '1234'):
            self.assertIsNone(floor.normalize_floor(source))


class TemporalTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        self.image, self.quad = fixture()
        self.ocr = fake_ocr()
        self.reader = floor.FloorDisplayReader(self.ocr)
        self.find = patch.object(floor, 'find_display', return_value=(self.quad, 'DISPLAY_FOUND')).start()
        self.addCleanup(patch.stopall)

    def process(self, at, image=None, **kwargs):
        return self.reader.process(self.image if image is None else image, at, **kwargs)

    def stable(self):
        for at in (1., 1.21, 1.42): result = self.process(at)
        return result

    def test_only_independent_ocr_observations_count_toward_stability(self):
        first = self.process(1.)
        for at in (1.03, 1.06, 1.09):
            cached = self.process(at)
            self.assertEqual(cached.independent_votes, 1)
            self.assertEqual(cached.source, 'PIXEL_VERIFIED_CACHE')
            self.assertIsNone(cached.current_floor)
        self.assertEqual(self.ocr.read.call_count, 1)
        second = self.process(1.21); third = self.process(1.42)
        self.assertEqual([first.state, second.state, third.state], ['UNSTABLE','UNSTABLE','STABLE'])
        self.assertEqual(third.current_floor, '6')
        self.assertEqual(self.ocr.read.call_count, 3)
        crop = self.ocr.read.call_args.args[0]
        self.assertLess(np.prod(crop.shape[:2]), .02*np.prod(self.image.shape[:2]))

    def test_changed_content_cannot_use_stable_old_floor_as_current(self):
        self.stable()
        changed, _ = fixture('3'); self.ocr.read.return_value = (('3', .98), 1.)
        pending = self.process(1.45, changed)
        self.assertIsNone(pending.current_floor)
        self.assertEqual(pending.last_confirmed_floor, '6')
        first = self.process(1.65, changed)
        self.assertNotEqual(first.state, 'STABLE')
        self.process(1.86, changed)
        result = self.process(2.07, changed)
        self.assertEqual(result.current_floor, '3')

    def test_changed_pixels_require_new_ocr_but_same_text_keeps_consensus(self):
        self.stable()
        actual = floor.text_region
        def changed(screen):
            patch_image, box, signature = actual(screen)
            return patch_image, box, 1-signature
        with patch.object(floor, 'text_region', side_effect=changed):
            pending = self.process(1.45)
            self.assertIsNone(pending.current_floor)
            self.assertNotEqual(pending.source, 'PIXEL_VERIFIED_CACHE')
            result = self.process(1.65)
        self.assertEqual(result.current_floor, '6')
        self.assertEqual(result.independent_votes, 3)

    def test_blank_lcd_or_lost_panel_has_no_current_floor(self):
        self.stable()
        blank, _ = fixture('')
        result = self.process(1.45, blank)
        self.assertEqual(result.state, 'NO_TEXT'); self.assertIsNone(result.current_floor)
        self.assertEqual(result.last_confirmed_floor, '6')
        lost = self.process(1.48, np.full_like(self.image, 170))
        self.assertIsNone(lost.current_floor); self.assertIsNone(lost.observed_floor)
        result = self.process(4., blank)
        self.assertIsNone(result.last_confirmed_floor)

    def test_duplicate_gap_generation_and_single_photo_do_not_inherit_stability(self):
        self.stable()
        self.assertEqual(self.process(1.42).state, 'STALE_FRAME')
        self.assertIsNone(self.process(1.63).current_floor)
        gap = self.process(3.)
        self.assertEqual(gap.independent_votes, 1)
        changed = self.process(3.03, generation=1)
        self.assertEqual(changed.independent_votes, 1)
        single = self.process(3.03, offline=True)
        self.assertEqual(single.state, 'SINGLE_SHOT'); self.assertIsNone(single.current_floor)

    def test_low_confidence_and_non_floor_text_do_not_create_floor(self):
        for raw, score, state in [('6', .79, 'LOW_CONFIDENCE'), ('V5', .99, 'NON_FLOOR_TEXT')]:
            self.reader.reset(); self.ocr.read.return_value = ((raw, score), 1.)
            result = self.process(1.)
            self.assertEqual(result.state, state)
            self.assertIsNone(result.observed_floor); self.assertIsNone(result.current_floor)
            self.assertEqual(result.raw_text, raw)

    def test_ocr_failure_has_bounded_retry_frequency_and_recovers(self):
        self.ocr.read.side_effect = RuntimeError('synthetic failure')
        result = self.process(1.)
        self.assertEqual(result.state, 'OCR_ERROR')
        for at in (1.01, 1.03, 1.06, 1.1):
            self.assertIsNone(self.process(at).current_floor)
        self.assertEqual(self.ocr.read.call_count, 1)
        self.ocr.read.side_effect = None
        self.assertEqual(self.process(1.21).observed_floor, '6')

    def test_multiple_displays_cannot_reappear_from_tracking_cache(self):
        self.stable(); self.find.return_value = (None, 'MULTIPLE_DISPLAYS')
        result = self.process(2.6)
        self.assertEqual(result.state, 'MULTIPLE_DISPLAYS')
        self.assertIsNone(result.current_floor)
        result = self.process(2.63)
        self.assertIsNone(result.observed_floor); self.assertIsNone(result.current_floor)

    def test_invalid_frame_timestamp_and_settings(self):
        for image in (None, np.zeros((50, 50), np.uint8), np.zeros((40, 40, 3), float)):
            result = self.reader.process(image, 1.)
            self.assertEqual(result.state, 'INVALID_FRAME')
        for at in (float('nan'), float('inf'), 'invalid'):
            result = self.process(at)
            self.assertEqual(result.state, 'INVALID_TIMESTAMP')
            json.dumps(result.to_dict(), allow_nan=False)
        for kwargs in ({'min_confidence':.1}, {'required':1}, {'interval_s':1.}, {'refresh_s':float('nan')}, {'max_side':100}):
            with self.assertRaises(ValueError): floor.FloorDisplayReader(self.ocr, **kwargs)


class OCRAndEntryTests(unittest.TestCase):
    def ocr(self, outputs):
        engine = NS(text_rec=Mock(side_effect=[NS(txts=[t], scores=[c]) for t, c in outputs]))
        with patch.object(floor, 'build_ocr', return_value=(engine, {'actual_device':'cpu','actual_backend':'FAKE_TEST'})) as build:
            reader = floor.DisplayOCR(device='cpu')
        build.assert_called_once()
        return reader, engine

    def test_one_crop_fast_path_and_one_retry_only_for_weak_text(self):
        reader, engine = self.ocr([('6',.99), ('V5',.999), ('5',.98)])
        patch_image = np.full((50,40,3),45,np.uint8)
        self.assertEqual(reader.read(patch_image)[0], ('6',.99))
        self.assertEqual(engine.text_rec.call_count, 1)
        self.assertEqual(reader.read(patch_image)[0], ('5',.98))
        self.assertEqual(engine.text_rec.call_count, 3)

    def test_bad_ocr_output_structure_and_score_are_reported(self):
        for raw in (NS(txts=None,scores=None), NS(txts=['6'],scores=[]), NS(txts=['6'],scores=[float('nan')])):
            reader, engine = self.ocr([]); engine.text_rec.side_effect = None; engine.text_rec.return_value = raw
            with self.assertRaises(RuntimeError): reader.read(np.zeros((48,48,3),np.uint8))

    def test_strict_cuda_failure_does_not_open_camera_or_fallback_silently(self):
        with patch.object(entry, 'DisplayOCR', side_effect=RuntimeError('cuda unavailable')), \
                patch.object(entry, 'run_camera') as camera, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(['--camera','--ocr-device','cuda']),2)
        camera.assert_not_called()

    def test_invalid_options_and_robot_flags_are_rejected(self):
        for argv in (['--execute'], ['--head-up'], ['--ocr-device','gpu'], ['--threads','0'],
                     ['--stable-reads','1'], ['--max-age','.1'], ['--roi','4','4','2','2']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): entry.parse_args(argv)

    def setup_camera(self):
        args = entry.parse_args(['--camera'])
        image = np.full((100,120,3),70,np.uint8)
        camera = Mock(); camera.get_frame.return_value = (image, None)
        reader = Mock(); reader.process.side_effect = lambda im, at, **kw: floor.DisplayResult(99.,im.shape[:2],generation=kw['generation'])
        pool = Mock(); pool.submit.side_effect = lambda fn: NS(done=lambda:True,result=fn)
        return args, image, camera, reader, pool

    def test_camera_reset_drops_old_epoch_and_does_not_save_frames(self):
        args, image, camera, reader, pool = self.setup_camera()
        with patch('drivers.camera.make_camera',return_value=camera), patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(entry.time,'perf_counter',return_value=99.2),patch.object(entry,'draw_result',return_value=image.copy()) as draw, \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'getWindowProperty',return_value=1), \
                patch.object(cv2,'waitKey',side_effect=[ord('r'),-1,ord('q')]),patch.object(cv2,'destroyAllWindows'), \
                patch.object(cv2,'imwrite',side_effect=AssertionError('no frame saving')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.run_camera(reader,args),0)
        draw.assert_called_once(); camera.close.assert_called_once(); pool.shutdown.assert_called_once()
        self.assertEqual([c.kwargs['generation'] for c in reader.process.call_args_list][:2],[0,1])

    def test_missing_frames_and_processing_failure_release_camera(self):
        args, _, camera, reader, pool = self.setup_camera(); camera.get_frame.return_value = None
        with patch('drivers.camera.make_camera',return_value=camera),patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',return_value=-1), \
                patch.object(cv2,'destroyAllWindows'),self.assertRaisesRegex(RuntimeError,'10次'):
            entry.run_camera(reader,args)
        self.assertEqual(camera.get_frame.call_count,10); camera.close.assert_called_once(); pool.shutdown.assert_called_once()
        args, _, camera, reader, pool = self.setup_camera(); reader.process.side_effect = RuntimeError('processing failure')
        with patch('drivers.camera.make_camera',return_value=camera),patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',return_value=-1), \
                patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyAllWindows'), \
                self.assertRaisesRegex(RuntimeError,'processing failure'):
            entry.run_camera(reader,args)
        camera.close.assert_called_once(); pool.shutdown.assert_called_once()

    def test_expired_result_not_drawn(self):
        args, _, camera, reader, pool = self.setup_camera()
        with patch('drivers.camera.make_camera',return_value=camera),patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(entry.time,'perf_counter',return_value=110.),patch.object(entry,'draw_result') as draw, \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',side_effect=[-1,ord('q')]), \
                patch.object(cv2,'getWindowProperty',return_value=1),patch.object(cv2,'destroyAllWindows'):
            entry.run_camera(reader,args)
        draw.assert_not_called(); camera.close.assert_called_once()

    def test_new_branch_has_no_robot_imports_or_control_calls(self):
        for path in (ENTRY, ROOT/'elevator/floor_display_261113.py'):
            tree=ast.parse(path.read_text())
            imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(n.startswith(('drivers.robot','elevator.app','elevator.motion','motorbridge')) for n in imports))
            calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            self.assertFalse(calls & {'connect','enable','move_to_traj','send_mit','set_zero_position','head_up'})


if __name__ == '__main__': unittest.main()
