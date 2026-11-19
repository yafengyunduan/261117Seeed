"""No devices: real sequence orchestration, original grasp steps, fake IO."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np
import yaml
from elevator import app,grasp,geometry,motion,perception,panel_261112
from utils.transforms import pose6d_to_mat4

ROOT=Path(__file__).resolve().parents[1]

def entry():
    spec=importlib.util.spec_from_file_location("sequence_entry",ROOT/"scripts/261110_my_grasp_then_elevator.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class SequenceTests(unittest.TestCase):
    def test_yaml_placement_and_cli_override_are_used_by_runtime_and_simulation(self):
        options=app.read_yaml(ROOT/"config/261105_my_elevator.yaml")
        options["grasp_place"]={"xyz_m":[.28,-.32,.36]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"workflow.yaml";path.write_text(yaml.safe_dump(options))
            original=path.read_bytes()
            for extra,expected in (([],[.28,-.32,.36]),
                                   (["--place-xyz-m",".3","-.25",".4"],[.3,-.25,.4])):
                argv=["--workflow-config",str(path),"--floor","3",*extra]
                with self.subTest(extra=extra),patch.object(app,"run_elevator",return_value=0) as run:
                    self.assertEqual(entry().main(argv),0)
                    self.assertEqual(run.call_args.args[2].place_xyz_m,expected)
                with patch.object(app,"run_elevator",side_effect=AssertionError("hardware")),contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(entry().main([*argv,"--simulate"]),0)
                np.testing.assert_allclose(np.asarray(json.loads(output.getvalue())["place_pose"])[:3,3],expected)
            self.assertEqual(path.read_bytes(),original)

    def test_entry_defaults_preview_and_uses_current_config_without_rewriting(self):
        with patch.object(app,"run_elevator",return_value=0) as run:
            self.assertEqual(entry().main(["--floor","3"]),0)
        cfg,options,args=run.call_args.args
        self.assertFalse(args.execute);self.assertTrue(args.sequence_mode)
        self.assertEqual((args.stage,args.floor),("contact","3"))
        self.assertEqual(args.place_xyz_m,options['grasp_place']['xyz_m'])
        self.assertEqual(options["elevator"]["target_offset_base_m"],app.read_yaml(ROOT/"config/261105_my_elevator.yaml")["elevator"]["target_offset_base_m"])

    def test_simulation_never_creates_devices(self):
        with patch.object(app,"run_elevator",side_effect=AssertionError("hardware")),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(entry().main(["--simulate","--floor","12"]),0)
        plan=json.loads(output.getvalue());self.assertTrue(plan["simulation_only"])
        self.assertEqual(plan['floor'],'12');self.assertIn('稳定识别12',plan['steps'])
        np.testing.assert_allclose(np.asarray(plan["place_pose"])[:3,3],app.load_workflow(ROOT/'config/261105_my_elevator.yaml')['grasp_place']['xyz_m'])

    def test_invalid_targets_or_times_fail_before_devices(self):
        for argv in (["--place-xyz-m","nan","0","0"],["--place-duration-s","0"],["--auto-floor-timeout-s","nan"],
                     ["--place-rpy-deg","0","inf","0"],["--standoff-m",".3"]):
            with self.subTest(argv=argv),patch.object(app,"run_elevator",side_effect=AssertionError("hardware")),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(entry().main(["--floor","3",*argv]),2)

    def test_floor_is_required_and_validated_before_config_or_devices(self):
        for argv in ([],['--execute'],['--floor','abc'],['--floor','3.5'],['--floor','100']):
            with self.subTest(argv=argv),patch.object(app,'read_yaml',side_effect=AssertionError('config')), \
                    patch.object(app,'run_elevator',side_effect=AssertionError('hardware')), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                entry().main(argv)
            self.assertEqual(error.exception.code,2)

    def test_explicit_floor_is_normalized_and_forwarded_without_overwrite(self):
        for supplied,expected in (('3','3'),('4','4'),('-1','-1'),('12','12'),('０３','3'),('−2','-2')):
            with self.subTest(floor=supplied),patch.object(app,'run_elevator',return_value=0) as run:
                self.assertEqual(entry().main(['--floor',supplied,'--execute']),0)
            self.assertEqual(run.call_args.args[2].floor,expected)

    def test_place_is_base_xyz_and_yaw_facing_point_with_optional_orientation(self):
        cfg={"robot":{"ready_pose":{"roll":0.,"pitch":.7}}}
        target=grasp.placement_pose(cfg,[.25,-.38,.35])
        expected=pose6d_to_mat4(.25,-.38,.35,0,.7,np.arctan2(-.38,.25))
        np.testing.assert_allclose(target,expected)
        np.testing.assert_allclose(grasp.placement_pose(cfg,[.25,-.38,.35],[0,0,0])[:3,:3],np.eye(3))

    def flow(self,*,execute=True,empty=False,fail_place=False,fail_release=False,fail_switch=False,
             missing_floor=False,invalid_depth=False,cancel_contact=False,abort_grasp=False,floor='3'):
        clock=[100.];events=[];moving=[False];g_sent=[False];g_at=[None];picked_snapshot=[]
        cfg={"camera":{"type":"realsense_d405"},"robot":{"ready_pose":dict(x=.3,y=0.,z=.3,roll=0.,pitch=.7,yaw=0.,duration=3.)}}
        options=app.load_workflow(ROOT/"config/261105_my_elevator.yaml")
        options["elevator"].update(dwell_s=0.,standoff_m=.05,target_offset_base_m=[.05,.0235,.0139],
                                   button_orientation='free')
        options['elevator']['refinement']['enabled']=False
        args=NS(execute=execute,stage="contact",device="cpu",floor=floor,sequence_mode=True,auto_floor_timeout_s=3.,
                place_xyz_m=[.25,-.38,.35],place_rpy_deg=None,place_duration_s=3.,release_wait_s=0.)
        class Camera:
            K=np.array([[100.,0,50],[0,100.,40],[0,0,1]]);D=np.zeros(5)
            def __init__(self):self.count=0;self.closed=False;self.opens=0
            def open(self):self.opens+=1
            def warm_up(self,n):pass
            def get_frame(self):
                self.count+=1;clock[0]+=.1
                if self.count>100:raise AssertionError("unbounded loop")
                d=0 if invalid_depth and worker.kind=="panel" else 400
                return np.zeros((80,100,3),np.uint8),np.full((80,100),d,np.uint16)
            def close(self):self.closed=True
        class Worker:
            def __init__(self):self.pending=None;self.closed=False;self.kind=None;self.panel_count=0
            def wait_ready(self,**kw):pass
            def submit(self,s,kind="panel"):self.pending=s;self.kind=kind;events.append("infer:"+kind)
            def receive(self):
                if self.pending is None:return None
                s,self.pending=self.pending,None
                if self.kind=="grasp":
                    picked_snapshot.append(s)
                    return s,[NS(is_valid=True,conf=.99,class_name="yellow banana",position=np.array([.02,.01,.3]),tcp_rotation=np.eye(3))]
                self.panel_count+=1
                distractor='4' if floor=='3' else '3'
                label=distractor if missing_floor or self.panel_count==1 else floor
                status="UNSTABLE" if self.panel_count==2 else "STABLE"
                b=perception.ButtonObservation(perception.Ring(50.,40.,12.,1.),label,.99,(48,38,52,42),status=status)
                return s,perception.FrameResult(s.captured_at,(80,100),(5,5,95,75),"YOLOE",[b])
            def close(self):self.closed=True
        class Driver:
            _application_cancelled=None
            def __init__(self):self.opens=0
            def open_gripper(self):self.opens+=1;events.append("open")
            def _position_done(self):return not (fail_release and self.opens==2)
            def grasp(self):events.append("grasp");return not empty
        class Robot:
            def __init__(self):self.T=pose6d_to_mat4(.26,0,.192,0,0,0);self.driver=Driver();self.moves=[];self.finished=False;self.head=False
            def start(self,**kw):events.append("start:"+str(kw.get("gripper_posvel")))
            def pose(self):return self.T.copy()
            def move(self,T,duration,cancelled):
                moving[0]=True
                try:
                    if cancelled():raise motion.MotionCancelled("cancel")
                    if fail_place and np.allclose(T[:3,3],args.place_xyz_m):raise RuntimeError("place IK failed")
                    self.T=T.copy();self.moves.append(T.copy());events.append("move:"+("panel" if self.head else "grasp/place"))
                finally:moving[0]=False
            def move_position(self,tip,duration,cancelled,*,tip_offset):
                target=geometry.pose_for_tip(self.T,tip,tip_offset)
                self.move(target,duration,cancelled)
                return target
            def switch_gripper_to_posvel(self,cancelled):
                events.append("switch")
                if fail_switch:raise RuntimeError("mode ack failed")
            def head_up(self,*a,**kw):
                self.head=True;events.append("head")
                self.head_target=list(a[0])
                self.T=pose6d_to_mat4(.26,0,.35,0,-.5,0)
            def stop(self):events.append("stop")
            def finish_task(self,*,home):self.finished=True;events.append("finish:"+str(home))
        camera=Camera();worker=Worker();robot=Robot()
        def key(*_):
            if abort_grasp and worker.kind=="grasp":return ord("q")
            if empty and "grasp" in events and not moving[0]:return ord("q")
            if cancel_contact and robot.head and moving[0]:return 32
            if worker.kind=="grasp" and not g_sent[0] and not moving[0]:
                g_sent[0]=True;g_at[0]=clock[0];return ord("g")
            return -1
        tracker=NS(panel=NS(stable=False),remembered_associations=lambda *a,**kw:(),cached_labels=lambda **kw:())
        tracker.update=lambda *a,**kw:setattr(tracker.panel,"stable",True)
        tracker.reset=lambda:setattr(tracker.panel,"stable",False)
        original=grasp.original_main
        def original_without_display():
            module=original()
            module._render_display=lambda image,*a:image.copy()
            module._print_best_grasp=lambda *a:None
            return module
        def before(robot,camera,worker,window):grasp.grasp_then_place(robot,camera,worker,window,cfg,args)
        with contextlib.ExitStack() as s:
            s.enter_context(patch("drivers.camera.make_camera",return_value=camera))
            s.enter_context(patch("utils.camera_utils.load_hand_eye",return_value=(np.eye(4),"eye_in_hand")))
            s.enter_context(patch.object(app,"PerceptionWorker",return_value=worker))
            factory=s.enter_context(patch.object(motion,"Robot",return_value=robot))
            s.enter_context(patch.object(app,"_tracker",return_value=tracker))
            s.enter_context(patch.object(grasp,"original_main",side_effect=original_without_display))
            s.enter_context(patch.object(perception,"draw_result",side_effect=lambda c,*a,**kw:c.copy()))
            for name in ("namedWindow","imshow","destroyAllWindows"):s.enter_context(patch("cv2."+name))
            s.enter_context(patch("cv2.waitKey",side_effect=key));s.enter_context(patch("cv2.getWindowProperty",return_value=1))
            s.enter_context(patch.object(app.sys.stdin,"isatty",return_value=False))
            s.enter_context(patch.object(app.time,"perf_counter",side_effect=lambda:clock[0]))
            s.enter_context(patch.object(app.time,"monotonic",side_effect=lambda:clock[0]))
            if fail_place or fail_switch:
                with self.assertRaisesRegex(RuntimeError,"place IK failed|mode ack failed"):
                    app.run_elevator(cfg,options,args,before_head=before)
            elif fail_release or missing_floor or invalid_depth:
                with self.assertRaises(TimeoutError) as error:app.run_elevator(cfg,options,args,before_head=before)
                if missing_floor:self.assertIn(f'等待稳定{floor}楼及有效深度超时',str(error.exception))
            else:
                status=app.run_elevator(cfg,options,args,before_head=before)
                self.assertEqual(status,130 if empty or cancel_contact or abort_grasp else 0)
        self.assertTrue(camera.closed and worker.closed)
        self.assertEqual(camera.opens,1)
        if execute:
            factory.assert_called_once();self.assertTrue(robot.finished)
            self.assertEqual(factory.call_args.args[0]['trajectory'],options['trajectory'])
        else:factory.assert_not_called()
        return events,robot,worker,g_at[0],picked_snapshot

    def test_complete_shared_session_uses_fresh_grasp_frame_then_release_then_stable_three_once(self):
        events,r,w,at,snapshots=self.flow()
        self.assertEqual(events[0],"start:False")
        self.assertEqual(events.count("grasp"),1);self.assertEqual(events.count("head"),1)
        self.assertEqual(r.head_target[-1],app.load_workflow(ROOT/'config/261105_my_elevator.yaml')['gripper']['closed_angle_deg'])
        self.assertEqual(events.count("open"),2)
        self.assertLess(events.index("grasp"),events.index("switch"))
        self.assertLess(max(i for i,e in enumerate(events) if e=="open"),events.index("switch"))
        self.assertLess(events.index("switch"),events.index("head"))
        self.assertGreater(snapshots[-1].captured_at,at)
        np.testing.assert_allclose(r.moves[4][:3,3],[.25,-.38,.35])
        self.assertEqual(len(r.moves),8) # ready, pre, grasp, ready, place, approach, surface, retreat
        self.assertGreaterEqual(w.panel_count,3)
        np.testing.assert_allclose(r.moves[-3][:3,:3],pose6d_to_mat4(0,0,0,0,-.5,0)[:3,:3])
        np.testing.assert_allclose(r.moves[-1],r.moves[-3])
        self.assertEqual(events[-1],"finish:True")

    def test_empty_grasp_or_user_abort_never_switches_to_elevator(self):
        for kwargs in (dict(empty=True),dict(abort_grasp=True)):
            with self.subTest(kwargs=kwargs):
                events,r,*_=self.flow(**kwargs)
                self.assertNotIn("switch",events);self.assertNotIn("head",events)
                self.assertEqual(events[-1],"finish:False")

    def test_nondefault_floor_runs_same_sequence_after_that_floor_becomes_stable(self):
        for floor in ('4','-1','12'):
            with self.subTest(floor=floor),contextlib.redirect_stdout(io.StringIO()) as output:
                events,r,w,*_=self.flow(floor=floor)
            self.assertIn(f'[Target] floor={floor} ',output.getvalue())
            self.assertIn(f'自动等待稳定{floor}楼',output.getvalue())
            self.assertIn(f'{floor}楼纸面接触及撤回完成',output.getvalue())
            self.assertGreaterEqual(w.panel_count,3)
            self.assertEqual(len(r.moves),8);self.assertEqual(events[-1],'finish:True')

    def test_stable_three_never_substitutes_for_missing_requested_floor(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            events,r,*_=self.flow(floor='4',missing_floor=True)
        self.assertEqual(events.count('move:panel'),0)
        self.assertEqual(events[-1],'finish:True')

    def test_place_release_and_mode_failures_stop_before_head_and_contact(self):
        for kwargs in (dict(fail_place=True),dict(fail_release=True),dict(fail_switch=True)):
            with self.subTest(kwargs=kwargs):
                events,r,*_=self.flow(**kwargs)
                self.assertNotIn("head",events)
                self.assertEqual(events[-1],"finish:"+str(bool(kwargs.get("fail_release"))))
                if kwargs.get("fail_place"):self.assertEqual(events.count("open"),1)

    def test_missing_three_invalid_depth_and_contact_cancel_never_touch_surface(self):
        for kwargs in (dict(missing_floor=True),dict(invalid_depth=True),dict(cancel_contact=True)):
            with self.subTest(kwargs=kwargs):
                events,r,*_=self.flow(**kwargs)
                self.assertIn("head",events)
                self.assertEqual(events[-1],"finish:"+str(not kwargs.get("cancel_contact",False)))
                self.assertEqual(events.count("move:panel"),0)

    def test_dry_run_only_uses_camera_and_model(self):
        events,r,*_=self.flow(execute=False)
        self.assertFalse(r.moves);self.assertNotIn("grasp",events);self.assertNotIn("switch",events)


class SharedInferenceTests(unittest.TestCase):
    def test_one_detector_handles_grasp_rgbd_and_panel_without_reinitializing(self):
        from tests.test_geometry import snapshot
        snap=snapshot();requests=queue.Queue();connection=NS(send=Mock(),close=Mock())
        requests.put(("grasp",1,1.,snap.color,snap.depth_mm,snap.K))
        requests.put(("panel",2,2.,snap.color,None,None));requests.put(None)
        detector=NS(ocr=NS(backend_info={}),warmup=Mock(),model=NS(predict=Mock(return_value=["yolo"])),
                    yolo_options={"device":"cpu","conf":.5,"iou":.45},process=Mock(return_value="panel_result"),request_refresh=Mock())
        with patch.object(panel_261112,"FastPanelPerception",return_value=detector) as construct, \
                patch("utils.ordinary_grasp.estimate_grasps",return_value="grasp_result") as estimate:
            app._perception_worker(connection,requests,{"camera":{}},{})
        construct.assert_called_once();detector.model.predict.assert_called_once();detector.process.assert_called_once()
        self.assertTrue(construct.call_args.args[1].workflow_evidence)
        detector.request_refresh.assert_called_once()
        estimate.assert_called_once();self.assertIs(estimate.call_args.args[1],snap.depth_mm)
        self.assertIs(estimate.call_args.args[2],snap.K)
        self.assertEqual([c.args[0] for c in connection.send.call_args_list],[("ready",),("result",1,"grasp_result"),("result",2,"panel_result")])

    def test_worker_request_carries_depth_only_for_grasp_and_one_frame_in_flight(self):
        from tests.test_geometry import snapshot
        worker=app.PerceptionWorker.__new__(app.PerceptionWorker);worker.pending=None;worker.requests=NS(put_nowait=Mock())
        snap=snapshot();worker.submit(snap,kind="grasp");worker.submit(snap,kind="panel")
        worker.requests.put_nowait.assert_called_once()
        self.assertIs(worker.requests.put_nowait.call_args.args[0][4],snap.depth_mm)
        worker.pending=None;worker.submit(snap)
        self.assertIsNone(worker.requests.put_nowait.call_args.args[0][4])
        self.assertEqual(worker.requests.put_nowait.call_args.args[0][6],snap.generation)

    def test_panel_generation_resets_shared_cache_without_loading_models_or_heading_again(self):
        from tests.test_geometry import snapshot
        snap=snapshot();requests=queue.Queue();connection=NS(send=Mock(),close=Mock())
        for i,generation in enumerate((1,1,2,3),1):
            requests.put(('panel',i,float(i),snap.color,None,None,generation))
        requests.put(None)
        detector=NS(ocr=NS(backend_info={}),warmup=Mock(),process=Mock(return_value='panel'),request_refresh=Mock())
        with patch.object(panel_261112,'FastPanelPerception',return_value=detector) as build:
            app._perception_worker(connection,requests,{'camera':{}},{})
        build.assert_called_once();detector.warmup.assert_called_once()
        self.assertEqual(detector.request_refresh.call_count,3)
        self.assertEqual(detector.process.call_count,4)


if __name__=="__main__":unittest.main()
