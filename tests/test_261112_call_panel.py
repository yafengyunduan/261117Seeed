"""Offline call-panel checks. Camera/UI are mocked; no robot/model startup."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np
from elevator import call_panel_261112 as call

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'scripts/261112_my_call_panel.py'
spec=importlib.util.spec_from_file_location('call_preview',ENTRY)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def rim(x,y,r=25.,support=1.):return call.vision.Ring(float(x),float(y),r,support)


class PairSelectionTests(unittest.TestCase):
    def test_vertical_order_is_used_even_if_input_is_reversed(self):
        pair,reason,_=call.select_pair([rim(200,260),rim(200,160)],(500,400))
        self.assertEqual(reason,'PAIR_FOUND');self.assertEqual([r.center for r in pair],[(200,160),(200,260)])

    def test_pair_rejects_single_horizontal_wrong_size_gap_tilt_or_clipped_rim(self):
        for rings in ([rim(200,160)],[rim(120,160),rim(220,160)],
                      [rim(200,160,25),rim(200,260,10)],[rim(200,160),rim(200,210)],
                      [rim(200,100),rim(200,350)],[rim(100,100),rim(160,200)],
                      [rim(10,160),rim(10,260)],[rim(200,160),rim(200,260,support=.4)]):
            with self.subTest(rings=rings):self.assertIsNone(call.select_pair(rings,(500,400))[0])

    def test_extra_buttons_or_multiple_panels_do_not_choose_one_by_score(self):
        pairs=([rim(200,y) for y in (100,200,300)],
               [rim(x,y) for y in (160,260) for x in (100,180)],
               [rim(x,y) for y in (160,260) for x in (100,330)])
        for rings in pairs:
            with self.subTest(rings=rings):self.assertIsNone(call.select_pair(rings,(500,500))[0])
        pair,reason,_=call.select_pair(pairs[-1],(500,500));self.assertEqual(reason,'AMBIGUOUS')

    def test_roi_can_disambiguate_two_separated_panels_but_not_hide_nearby_third(self):
        rings=[rim(x,y) for y in (160,260) for x in (100,330)]
        pair,_,_=call.select_pair(rings,(500,500),(60,100,140,320))
        self.assertEqual([r.center for r in pair],[(100,160),(100,260)])
        self.assertIsNone(call.select_pair([rim(200,y) for y in (100,200,300)],(500,500),(150,50,250,250))[0])

    def test_nested_duplicates_are_not_extra_buttons(self):
        pair,_,_=call.select_pair([rim(200,160),rim(201,160,24,.9),rim(200,260)],(500,400))
        self.assertIsNotNone(pair)

    def test_invalid_geometry_is_not_a_pair(self):
        for bad in (rim(float('nan'),160),rim(200,float('inf')),rim(200,160,float('nan'))):
            self.assertIsNone(call.select_pair([bad,rim(200,260)],(500,400))[0])


class TrackingTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        self.rings=[rim(200,160),rim(200,260)]
        self.image=np.full((500,400,3),90,np.uint8)
        for r in self.rings:
            cv2.circle(self.image,r.center,25,(230,230,230),3)
            cv2.line(self.image,(int(r.x)-10,int(r.y)),(int(r.x),int(r.y)-10),(150,150,150),2)
        self.detector=call.CallPanelDetector()

    def process(self,at,im=None,**kwargs):
        return self.detector.process(self.image if im is None else im,at,**kwargs)

    def prime(self):
        for at in (1.,1.03,1.06):result=self.process(at)
        return result

    def test_three_fresh_frames_stabilize_with_only_one_full_search(self):
        with patch.object(call.vision,'find_rings',return_value=self.rings) as find:
            first=self.process(1.);second=self.process(1.03);third=self.process(1.06)
            for k in range(20):last=self.process(1.09+k*.03)
        self.assertEqual([first.state,second.state,third.state],['UNSTABLE','UNSTABLE','STABLE'])
        self.assertEqual(last.state,'STABLE');find.assert_called_once()
        self.assertTrue(last.geometry_source.startswith('LOCAL_'))
        x0,y0,x1,y1=last.tracking_roi
        self.assertLess((x1-x0)*(y1-y0)/(500*400),.2)

    def test_one_missing_rim_removes_both_labels_and_recovery_restarts_stability(self):
        with patch.object(call.vision,'find_rings',return_value=self.rings):
            self.prime();occluded=self.image.copy();occluded[220:300,160:240]=90
            lost=self.process(1.10,occluded)
            self.assertFalse(lost.pair);self.assertNotEqual(lost.state,'STABLE')
            recovered=self.process(1.13)
            self.assertEqual(recovered.state,'UNSTABLE')
            self.process(1.16);self.assertEqual(self.process(1.19).state,'STABLE')

    def test_duplicate_and_long_gap_or_generation_cannot_inherit_stable_status(self):
        with patch.object(call.vision,'find_rings',return_value=self.rings):
            self.prime();stale=self.process(1.06)
            self.assertEqual(stale.state,'STALE_FRAME');self.assertFalse(stale.pair)
            self.assertEqual(self.process(1.09).stable_observations,1)
            self.assertEqual(self.process(2.5).stable_observations,1)
            fresh=self.process(2.53,generation=1)
            self.assertEqual(fresh.state,'UNSTABLE');self.assertEqual(fresh.stable_observations,1)

    def test_context_change_detects_a_third_button_before_periodic_audit(self):
        changed=self.image.copy();cv2.circle(changed,(200,360),25,(230,230,230),3)
        with patch.object(call.vision,'find_rings',side_effect=[self.rings,self.rings+[rim(200,360)]]) as find:
            self.prime();result=self.process(1.1,changed)
        self.assertEqual(find.call_count,2)
        self.assertFalse(result.pair);self.assertEqual(result.state,'EXTRA_BUTTONS')

    def test_periodic_global_miss_keeps_two_current_pixel_verified_rims(self):
        with patch.object(call.vision,'find_rings',side_effect=[self.rings,[]]) as find:
            for i in range(24):result=self.process(1.+i*.1)
        self.assertEqual(find.call_count,2);self.assertEqual(result.state,'STABLE')
        self.assertEqual(len(result.pair),2)

    def test_simple_brightness_changes_do_not_force_repeated_global_search(self):
        with patch.object(call.vision,'find_rings',return_value=self.rings) as find:
            self.prime()
            for i in range(8):
                im=np.uint8(np.clip(self.image.astype(float)*(.8+.02*i)+10,0,255))
                result=self.process(1.09+i*.03,im)
        self.assertEqual(result.state,'STABLE');find.assert_called_once()

    def test_moving_pair_does_not_remain_stable(self):
        with patch.object(call.vision,'find_rings',return_value=self.rings):
            self.prime()
            moved=cv2.warpAffine(self.image,np.float32([[1,0,6],[0,1,0]]),(400,500))
            result=self.process(1.1,moved)
        self.assertNotEqual(result.state,'STABLE')

    def test_acquisition_is_throttled_and_resolution_escalates_on_misses(self):
        blank=np.zeros_like(self.image)
        with patch.object(call.vision,'find_rings',return_value=[]) as find:
            for at in (1.,1.03,1.1,1.4,1.8):self.process(at,blank)
        self.assertEqual(find.call_count,3)
        self.assertEqual([c.args[1] for c in find.call_args_list],[640,960,1200])

    def test_invalid_frames_or_timestamp_have_no_targets_and_valid_json(self):
        for im in (None,np.zeros((40,40),np.uint8),np.zeros((40,40,3),float)):
            result=self.detector.process(im,1.)
            self.assertEqual(result.state,'INVALID_FRAME');self.assertFalse(result.pair)
        for at in (float('nan'),float('inf'),'invalid'):
            result=self.detector.process(self.image,at)
            self.assertEqual(result.state,'INVALID_TIMESTAMP')
            json.dumps(result.to_dict(),allow_nan=False)

    def test_offline_images_are_single_shots_and_never_enable_actions(self):
        with patch.object(call.vision,'find_rings',return_value=self.rings):
            for _ in range(3):result=self.process(1.,offline=True)
        self.assertEqual(result.state,'SINGLE_SHOT');self.assertEqual(result.stable_observations,1)
        data=result.to_dict();self.assertFalse(data['actions_enabled'])
        self.assertEqual([b['direction'] for b in data['buttons']],['UP','DOWN'])
        self.assertTrue(all(b['direction_source']=='IMAGE_VERTICAL_ORDER' for b in data['buttons']))


class PhotoAndEntryTests(unittest.TestCase):
    def test_both_supplied_photos_work_without_any_model_or_ocr_initialization(self):
        expected={'电梯上下按键全.jpg':[(424,446),(426,575)],'电梯上下按键大.jpg':[(430,454),(431,781)]}
        cv2.setNumThreads(2)
        with patch.object(call.vision,'ButtonOCR',side_effect=AssertionError('no OCR')), \
                patch.object(call.vision,'load_yolo',side_effect=AssertionError('no YOLO')):
            detector=call.CallPanelDetector()
            for name,centers in expected.items():
                image=cv2.imread(str(ROOT/'readme/PairDrop_files_20261112_1513'/name))
                result=detector.process(image,offline=True)
                self.assertEqual(len(result.pair),2)
                for r,xy in zip(result.pair,centers):self.assertLess(np.linalg.norm(np.subtract(r.center,xy)),10.)

    def test_close_up_large_rim_fallback_and_multibutton_rejection(self):
        path=ROOT/'readme/PairDrop_files_20261112_1513/电梯上下按键大.jpg'
        image=cv2.imread(str(path));h,w=image.shape[:2]
        matrix=cv2.getRotationMatrix2D((w/2,h/2),0,1.2)
        enlarged=cv2.warpAffine(image,matrix,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        result=call.CallPanelDetector().process(enlarged,offline=True)
        self.assertEqual(len(result.pair),2)
        self.assertIn('LARGE_GRADIENT_ALT',result.acquisition_methods)
        for r,xy in zip(result.pair,[(430,454),(431,781)]):
            self.assertLess(np.linalg.norm(np.array(r.center)-matrix@np.array([*xy,1.])),12.)
        rings=[rim(200,y) for y in (100,200,300)]
        with patch.object(call.vision,'find_rings',return_value=rings),patch.object(call,'_large_rims') as fallback:
            result=call.CallPanelDetector().process(image,offline=True)
        self.assertFalse(result.pair);fallback.assert_not_called()

    def test_invalid_options_and_motion_flags_are_rejected(self):
        for argv in (['--execute'],['--head-up'],['--threads','0'],['--max-side','1'],
                     ['--stable-frames','2'],['--refresh-s','nan'],['--roi','4','4','2','2']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):entry.parse_args(argv)

    def camera_setup(self):
        args=entry.parse_args(['--camera']);image=np.full((100,120,3),70,np.uint8)
        camera=Mock();camera.get_frame.return_value=(image,None)
        detector=Mock();detector.process.side_effect=lambda im,at,**kw:call.CallPanelResult(99.,im.shape[:2],generation=kw['generation'])
        pool=Mock();pool.submit.side_effect=lambda fn:NS(done=lambda:True,result=fn)
        return args,image,camera,detector,pool

    def test_camera_reset_discards_old_result_and_window_is_not_saved(self):
        args,image,camera,detector,pool=self.camera_setup()
        with patch('drivers.camera.make_camera',return_value=camera),patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(entry.time,'perf_counter',return_value=99.2),patch.object(entry,'draw_result',return_value=image.copy()) as draw, \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'getWindowProperty',return_value=1), \
                patch.object(cv2,'waitKey',side_effect=[ord('r'),-1,ord('q')]),patch.object(cv2,'destroyAllWindows'), \
                patch.object(cv2,'imwrite',side_effect=AssertionError('no per-frame saving')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.run_camera(detector,args),0)
        draw.assert_called_once();camera.close.assert_called_once();pool.shutdown.assert_called_once()
        self.assertEqual([c.kwargs['generation'] for c in detector.process.call_args_list][:2],[0,1])

    def test_camera_failures_are_bounded_and_cleanup_still_runs(self):
        args,_,camera,detector,pool=self.camera_setup();camera.get_frame.return_value=None
        with patch('drivers.camera.make_camera',return_value=camera),patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',return_value=-1), \
                patch.object(cv2,'destroyAllWindows'),self.assertRaisesRegex(RuntimeError,'10次'):
            entry.run_camera(detector,args)
        self.assertEqual(camera.get_frame.call_count,10);camera.close.assert_called_once();pool.shutdown.assert_called_once()

    def test_camera_expired_result_is_not_drawn(self):
        args,_,camera,detector,pool=self.camera_setup()
        with patch('drivers.camera.make_camera',return_value=camera),patch.object(entry,'ThreadPoolExecutor',return_value=pool), \
                patch.object(entry.time,'perf_counter',return_value=110.),patch.object(entry,'draw_result') as draw, \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'imshow'),patch.object(cv2,'getWindowProperty',return_value=1), \
                patch.object(cv2,'waitKey',side_effect=[-1,ord('q')]),patch.object(cv2,'destroyAllWindows'):
            entry.run_camera(detector,args)
        draw.assert_not_called();camera.close.assert_called_once()

    def test_no_robot_imports_control_calls_or_model_loading(self):
        for path in (ENTRY,ROOT/'elevator/call_panel_261112.py'):
            tree=ast.parse(path.read_text())
            names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(n.startswith(('elevator.app','elevator.motion','drivers.robot','motorbridge')) for n in names))
            calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            self.assertFalse(calls&{'connect','enable','move_to_traj','send_mit','set_zero_position','head_up'})


if __name__=='__main__':unittest.main()
