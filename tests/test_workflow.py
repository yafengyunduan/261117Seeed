"""Current simplified main.py-like workflow, all hardware mocked."""
from argparse import Namespace
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np
from elevator import app,geometry,motion,perception
from utils.transforms import pose6d_to_mat4

ROOT=Path(__file__).resolve().parents[1]

class WorkflowTests(unittest.TestCase):
    def settings(self):
        o=app.load_workflow(ROOT/"config/261105_my_elevator.yaml");o["elevator"]["dwell_s"]=0.
        # Fixed regression fixture; users may tune their live YAML independently.
        o["elevator"].update(standoff_m=.05,target_offset_base_m=[.05,.0235,.0139])
        return o

    def test_legacy_workflow_without_placement_keeps_original_xyz(self):
        o=self.settings();o.pop("grasp_place")
        with patch.object(app,"read_yaml",return_value=o):
            self.assertEqual(app.load_workflow("unused")["grasp_place"]["xyz_m"],[.25,-.38,.35])

    def test_invalid_shared_initial_poses_are_rejected_before_runtime(self):
        for key,value in (('grasp_ready',None),('grasp_ready',{'x':.3}),
                          ('call_view_deg',[0,0,-25]),('floor_view_deg',[0,0,float('nan'),0,0,0]),
                          ('display_view_deg',[0,0,-40,0,0,float('inf')])):
            o=self.settings();o['initial_poses'][key]=value
            with self.subTest(key=key,value=value),patch.object(app,'read_yaml',return_value=o),self.assertRaises(ValueError):
                app.load_workflow('unused')
        o=self.settings();o['initial_poses']['grasp_ready']['duration']=0.
        with patch.object(app,'read_yaml',return_value=o),self.assertRaises(ValueError):app.load_workflow('unused')
        for seconds in (-.1,float('nan'),True):
            o=self.settings();o['initial_poses']['grasp_ready']['joint4_stage_duration_s']=seconds
            with self.subTest(seconds=seconds),patch.object(app,'read_yaml',return_value=o), \
                    self.assertRaisesRegex(ValueError,'joint4_stage_duration_s'):
                app.load_workflow('unused')

    def test_floor_cache_retention_flag_is_boolean_and_old_configs_remain_timed(self):
        for value in (None,0,'true'):
            o=self.settings();o['elevator']['floor_cache_until_update']=value
            with patch.object(app,'read_yaml',return_value=o),self.assertRaises(ValueError):app.load_workflow('unused')
        o=self.settings();o['elevator'].pop('floor_cache_until_update')
        with patch.object(app,'read_yaml',return_value=o):
            self.assertFalse(app.load_workflow('unused')['elevator']['floor_cache_until_update'])

    def test_invalid_yaml_placement_is_rejected(self):
        for value in (None,[],{"xyz_m":[.25,-.38]},
                      {"xyz_m":[.25,float("nan"),.35]},
                      {"xyz_m":[.25,-.38,float("inf")]}):
            o=self.settings();o["grasp_place"]=value
            with self.subTest(value=value),patch.object(app,"read_yaml",return_value=o),self.assertRaises(ValueError):
                app.load_workflow("unused")

    def test_print_contact_is_surface_and_retreat_without_push(self):
        r=NS(move=Mock(),pose=lambda:np.eye(4));e=self.settings()["elevator"]
        poses=app.execute_button(r,np.array([.4,-.1,.4]),np.eye(4),e,[0,0,0],"contact")
        self.assertEqual(r.move.call_count,3)
        np.testing.assert_allclose(r.move.call_args_list[1].args[0],poses["surface"])
        np.testing.assert_allclose(r.move.call_args_list[-1].args[0],poses["retreat"])

    def test_approach_stops_before_surface(self):
        r=NS(move=Mock(),pose=lambda:np.eye(4));e=self.settings()["elevator"]
        app.execute_button(r,np.array([.4,-.1,.4]),np.eye(4),e,[0,0,0],"approach")
        self.assertEqual(r.move.call_count,1)
        with self.assertRaises(ValueError):app.execute_button(r,np.zeros(3),np.eye(4),e,[0,0,0],"observe")

    def test_ik_failure_prevents_surface_and_retreat_commands(self):
        r=NS(move=Mock(side_effect=RuntimeError("IK failed")))
        with self.assertRaises(RuntimeError):app.execute_button(r,np.array([.4,-.1,.4]),np.eye(4),self.settings()["elevator"],[0,0,0],"contact")
        self.assertEqual(r.move.call_count,1)

    def test_free_position_moves_keep_tip_targets_while_orientation_changes(self):
        initial=pose6d_to_mat4(.26,0.,.35,.1,-.7,.3);state=[initial.copy()]
        offset=np.array([.03,.01,.02]);point=np.array([.473,-.029,.734])
        goals=[]
        def move_position(target,duration,cancelled,*,tip_offset):
            goals.append(np.asarray(target).copy())
            T=pose6d_to_mat4(0.,0.,0.,.2,-.5-.1*len(goals),.15)
            T[:3,3]=np.asarray(target)-T[:3,:3]@np.asarray(tip_offset)
            state[0]=T
            return T
        robot=NS(move=Mock(),move_position=Mock(side_effect=move_position),pose=lambda:state[0].copy())
        settings=self.settings()['elevator'];settings['standoff_m']=.1
        poses=app.execute_button(robot,point,initial,settings,offset,'contact',position_only=True)
        direction=np.array([np.cos(.3),np.sin(.3),0.])
        self.assertEqual(len(goals),3);robot.move.assert_not_called()
        for actual,distance in zip(goals,(-.1,0.,-.1)):
            np.testing.assert_allclose(actual,point+direction*distance)
        for name,distance in (('approach',-.1),('surface',0.),('retreat',-.1)):
            T=poses[name]
            np.testing.assert_allclose(T[:3,3]+T[:3,:3]@offset,point+direction*distance)

    def test_printed_mode_rejects_invented_push_and_config_has_no_old_control_gates(self):
        o=self.settings()
        self.assertNotIn("motion",o);self.assertNotIn("execution",o)
        o["elevator"]["press_travel_m"]=.003
        with patch.object(app,"read_yaml",return_value=o),self.assertRaises(ValueError):
            app.load_workflow(Path("unused"))

    def test_simulation_and_mode_prompt_do_not_create_robot(self):
        import json
        output=io.StringIO()
        options=self.settings()
        options['elevator']['refinement'].update(enabled=True,vertical_panel=True)
        options['elevator']['standoff_m']=.20
        with patch.object(motion,"Robot",side_effect=AssertionError("hardware")),contextlib.redirect_stdout(output), \
                patch.object(app,"load_workflow",return_value=options):
            self.assertEqual(app.main(["--simulate","--floor","3"]),0)
        simulated=json.loads(output.getvalue())
        np.testing.assert_allclose(simulated["corrected_base_m"],[.45,-.0765,.4139])
        positions={key:np.asarray(pose)[:3,3] for key,pose in simulated['targets'].items()}
        approach=positions['approach']-positions['surface']
        self.assertAlmostEqual(np.linalg.norm(approach),.20)
        self.assertAlmostEqual(approach[2],0.)
        self.assertNotEqual(approach[1],0.)
        np.testing.assert_allclose(positions['retreat'],positions['approach'])
        with patch("builtins.input",return_value="2"),patch.object(app,"run_elevator",return_value=0) as run:
            self.assertEqual(app.main([]),0)
        self.assertFalse(run.call_args.args[2].execute)

    def test_startup_preview_limits_rgbd_reads_but_checks_quit_every_poll(self):
        color=np.zeros((10,10,3),np.uint8)
        camera=NS(get_frame=Mock(return_value=(color,None)))
        draw,keys=Mock(),Mock(side_effect=[False,False,False,False,True])
        with patch.object(app.time,'perf_counter',side_effect=[0.,.05,.10,.20,.25]):
            callback=app.startup_preview(camera,draw,keys)
            for _ in range(4):callback()
            with self.assertRaises(motion.MotionCancelled):callback()
        self.assertEqual(camera.get_frame.call_count,2)
        self.assertEqual(draw.call_count,2)
        self.assertEqual(keys.call_count,5)

    def test_check_bus_bypasses_mode_prompt_camera_and_elevator_configuration(self):
        with patch("builtins.input",side_effect=AssertionError("no prompt")), \
                patch.object(app,"run_elevator",side_effect=AssertionError("no camera")), \
                patch.object(app,"load_workflow",side_effect=AssertionError("no OCR config")), \
                patch.object(motion,"check_bus",return_value=2) as check:
            self.assertEqual(app.main(["--check-bus"]),2)
            check.assert_called_once()
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            app.parse_args(["--check-bus","--execute"])

    def runtime(self,execute=True,stage="approach",fail_move=False,cancel_move=False,fail_close=False,
                cache_age=2.,combined=True,head_failure=False,initialize_gripper=True,
                target_offset=(0.,0.,0.),base_compensation=(0.,0.,0.),button_orientation='level'):
        clock=[100.]
        class Camera:
            K=np.array([[100.,0,50],[0,100.,40],[0,0,1]]);D=np.zeros(5)
            def __init__(self):self.count=0;self.closed=False
            def open(self):pass
            def warm_up(self,n):pass
            def get_frame(self):
                self.count+=1
                clock[0]+=.2
                if self.count>25:raise AssertionError("loop never completed")
                return np.zeros((80,100,3),np.uint8),np.full((80,100),500+self.count,np.uint16)
            def close(self):self.closed=True
        class Worker:
            def __init__(self,*a,**kw):self.pending=None;self.closed=False
            def wait_ready(self,**kw):pass
            def submit(self,s):self.pending=s
            def receive(self):
                if self.pending is None:return None
                snap,self.pending=self.pending,None
                b=perception.ButtonObservation(perception.Ring(50.,40.,12.,1.),"3",.99,(1,1,3,3),status="STABLE")
                return snap,perception.FrameResult(snap.captured_at,(80,100),(5,5,95,75),"YOLOE",[b])
            def close(self):self.closed=True
        class Robot:
            def __init__(self,*a):
                self.T=pose6d_to_mat4(.26,0,.192,0,-.5,0)
                self.moves=[];self.position_moves=[];self.finished=False;self.homed=None
            def start(self):pass
            def head_up(self,*a,**kw):
                self.head_target=a[0]
                self.head_wait=kw["wait_for_arrival"]
                if head_failure and self.head_wait:raise TimeoutError("gripper not at zero")
            def pose(self):return self.T.copy()
            def move(self,T,*a):
                if fail_move:raise RuntimeError("IK failed")
                self.moves.append(np.array(T).copy());self.T=np.array(T).copy()
                if cancel_move and a[-1]():raise motion.MotionCancelled("user stop")
            def move_position(self,tip,duration,cancelled,*,tip_offset):
                self.position_moves.append(tip.copy())
                target=geometry.pose_for_tip(self.T,tip,tip_offset)
                self.move(target,duration,cancelled)
                return target
            def stop(self):pass
            def finish_task(self,*,home):
                self.finished=True;self.homed=home
                if fail_close:raise RuntimeError("hold unconfirmed")
        camera=Camera();worker=Worker();robot=Robot()
        tracker=NS(panel=NS(stable=False),remembered_associations=lambda r,now:(),cached_labels=lambda **kw:())
        tracker.update=lambda *a,**kw:setattr(tracker.panel,"stable",True)
        tracker.reset=lambda:setattr(tracker.panel,"stable",False)
        args=NS(execute=execute,stage=stage,device="cpu",floor="3")
        expected=1 if stage=="observe" else (1 if stage=="approach" else 3)+int(not combined and button_orientation=='level')
        cancelled_once=[False]
        def key(delay):
            if cancel_move and robot.moves:
                if not cancelled_once[0]:cancelled_once[0]=True;return 32
                return 27
            if (not execute and camera.count>=7) or (stage=="observe" and camera.count>=7) or len(robot.moves)>=expected:
                return ord("q")
            return -1
        cfg={"camera":{"type":"realsense_d405"},"calibration":{"hand_eye_compensation_m":dict(zip("xyz",base_compensation))}}
        options=self.settings();options["elevator"].update(stable_snapshot_age_s=cache_age,level_during_approach=combined,
                                                         target_offset_base_m=list(target_offset),button_orientation=button_orientation)
        options['elevator']['refinement']['enabled']=False
        options["gripper"]["initialize"]=initialize_gripper
        if head_failure:options["startup_view"]["wait_for_arrival"]=True
        with contextlib.ExitStack() as s:
            s.enter_context(patch("drivers.camera.make_camera",return_value=camera))
            s.enter_context(patch("utils.camera_utils.load_hand_eye",return_value=(np.eye(4),"eye_in_hand")))
            s.enter_context(patch.object(app,"PerceptionWorker",return_value=worker))
            s.enter_context(patch.object(app,"_tracker",return_value=tracker))
            factory=s.enter_context(patch.object(motion,"Robot",return_value=robot))
            s.enter_context(patch.object(perception,"draw_result",side_effect=lambda c,*a,**kw:c.copy()))
            for name in ("namedWindow","imshow","destroyAllWindows"):s.enter_context(patch("cv2."+name))
            s.enter_context(patch("cv2.waitKey",side_effect=key))
            s.enter_context(patch("cv2.getWindowProperty",return_value=1))
            s.enter_context(patch.object(app.sys.stdin,"isatty",return_value=False))
            s.enter_context(patch.object(app.time,"perf_counter",side_effect=lambda:clock[0]))
            if head_failure:
                with self.assertRaisesRegex(TimeoutError,"gripper not at zero"):app.run_elevator(cfg,options,args)
            elif fail_move:
                with self.assertRaisesRegex(RuntimeError,"IK failed"):app.run_elevator(cfg,options,args)
            elif fail_close:
                with self.assertRaisesRegex(RuntimeError,"退出清理未全部确认.*hold unconfirmed"):
                    app.run_elevator(cfg,options,args)
            else:self.assertEqual(app.run_elevator(cfg,options,args),0)
        return camera,worker,robot,factory

    def test_runtime_preview_never_creates_robot_and_releases_camera(self):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            camera,worker,robot,factory=self.runtime(execute=False,target_offset=(.05,.0235,.0139))
        factory.assert_not_called();self.assertTrue(camera.closed and worker.closed)
        self.assertIn("base_m=None",output.getvalue())

    def test_cached_and_new_rgbd_apply_original_and_elevator_correction_once_through_contact(self):
        correction=self.settings()["elevator"]["target_offset_base_m"]
        original=[-.02,-.02,-.05]
        for cache_age,depth,source in ((2.,.502,"STABLE_CACHE"),(0.,.503,"NEW_RESULT")):
            with self.subTest(source=source),contextlib.redirect_stdout(io.StringIO()) as output:
                _,_,robot,_=self.runtime(stage="contact",cache_age=cache_age,target_offset=correction,base_compensation=original)
            expected=(pose6d_to_mat4(.26,0,.192,0,-.5,0)@np.array([0,0,depth,1]))[:3]+original+np.array(correction)
            np.testing.assert_allclose(robot.moves[1][:3,3],expected)
            np.testing.assert_allclose(robot.moves[0][:3,3],expected+[-.05,0,0])
            np.testing.assert_allclose(robot.moves[2],robot.moves[0])
            self.assertIn("source="+source,output.getvalue())
            self.assertIn('"target_offset_base_m": [0.05, 0.0235, 0.0139]',output.getvalue())

    def test_complete_cached_snapshot_is_frozen_through_combined_approach_contact_retreat(self):
        camera,worker,robot,_=self.runtime(stage="contact")
        self.assertEqual(len(robot.moves),3)
        # Frame 2 was captured at floor input, before any later frame.
        # Use its depth AND original pose, not the current frame's depth.
        captured_pose=pose6d_to_mat4(.26,0,.192,0,-.5,0)
        expected=(captured_pose@np.array([0.,0.,.502,1.]))[:3]
        np.testing.assert_allclose(robot.moves[1][:3,3],expected,atol=1e-8)
        np.testing.assert_allclose(robot.moves[2],robot.moves[0])
        self.assertTrue(camera.closed and worker.closed and robot.finished and robot.homed)

    def test_single_solution_free_mode_reuses_one_target_with_no_extra_leveling(self):
        with patch.object(geometry,'point_from_button',wraps=geometry.point_from_button) as solve, \
                patch.object(geometry,'measure_button_surface',side_effect=AssertionError('plane fit')), \
                patch.object(app,'reacquire_button',side_effect=AssertionError('second localization')):
            _,_,robot,_=self.runtime(stage='contact',combined=False,button_orientation='free')
        solve.assert_called_once()
        captured_pose=pose6d_to_mat4(.26,0,.192,0,-.5,0)
        point=(captured_pose@np.array([0.,0.,.502,1.]))[:3]
        standoff=point-np.array([.05,0,0])
        np.testing.assert_allclose(robot.position_moves,[standoff,point,standoff])
        self.assertEqual(len(robot.moves),3)
        np.testing.assert_allclose(robot.moves[0][:3,:3],captured_pose[:3,:3])

    def test_disable_cache_and_combined_move_restores_post_input_frame_and_separate_level(self):
        _,_,robot,_=self.runtime(stage="contact",cache_age=0.,combined=False)
        self.assertEqual(len(robot.moves),4)
        captured_pose=pose6d_to_mat4(.26,0,.192,0,-.5,0)
        expected=(captured_pose@np.array([0.,0.,.503,1.]))[:3]
        np.testing.assert_allclose(robot.moves[2][:3,3],expected,atol=1e-8)
        np.testing.assert_allclose(robot.moves[3],robot.moves[1])

    def test_runtime_fault_finishes_without_automatic_home(self):
        camera,worker,robot,_=self.runtime(fail_move=True)
        self.assertTrue(camera.closed and worker.closed and robot.finished)
        self.assertFalse(robot.homed)

    def test_exit_failure_surfaces_and_other_resources_are_closed(self):
        camera,worker,robot,_=self.runtime(fail_close=True)
        self.assertTrue(camera.closed and worker.closed and robot.finished)

    def test_motion_error_is_preserved_when_cleanup_also_fails(self):
        camera,worker,robot,_=self.runtime(fail_move=True,fail_close=True)
        self.assertTrue(camera.closed and worker.closed and robot.finished)

    def test_user_stop_during_approach_never_continues_to_surface_or_homes(self):
        camera,worker,robot,_=self.runtime(stage="contact",cancel_move=True)
        self.assertEqual(len(robot.moves),1)
        self.assertTrue(camera.closed and worker.closed and robot.finished)
        self.assertFalse(robot.homed)

    def test_observe_does_not_level_or_approach(self):
        camera,worker,robot,_=self.runtime(stage="observe")
        self.assertEqual(robot.moves,[])
        options=self.settings()
        self.assertEqual(robot.head_target,options['initial_poses']['floor_view_deg']+[options['gripper']['closed_angle_deg']])
        self.assertFalse(robot.head_wait)

    def test_invalid_head_wait_or_compensation_rejected_before_devices(self):
        for value in ("false",0,None):
            o=self.settings();o["startup_view"]["wait_for_arrival"]=value
            with patch.object(app,"read_yaml",return_value=o),self.assertRaises(ValueError):app.load_workflow(Path("unused"))
        for value in ([0,0],[0,0,float("nan")]):
            o=self.settings();o["elevator"]["target_offset_base_m"]=value
            with patch.object(app,"read_yaml",return_value=o),self.assertRaises(ValueError):app.load_workflow(Path("unused"))

    def test_gripper_arrival_failure_prevents_approach_even_with_automatic_floor(self):
        camera,worker,robot,_=self.runtime(stage="contact",head_failure=True)
        self.assertEqual(robot.moves,[])
        self.assertTrue(camera.closed and worker.closed and robot.finished)
        self.assertTrue(robot.homed)

    def test_skip_gripper_sends_six_targets_only(self):
        _,_,robot,_=self.runtime(stage="observe",initialize_gripper=False)
        self.assertEqual(robot.head_target,self.settings()['initial_poses']['floor_view_deg'])

    def test_invalid_seventh_angle_is_rejected_before_devices(self):
        for last in (float("nan"),0.):
            o=self.settings();o["startup_view"]["joints_deg"]=[0,0,-10,-20,0,0,last]
            o.pop('initial_poses',None)  # Exercise the legacy seven-axis config format.
            o['gripper']['closed_angle_deg']=10.
            with patch.object(app,"read_yaml",return_value=o),self.assertRaises(ValueError):
                app.load_workflow(Path("unused"))

    def test_gripper_reference_is_validated_and_legacy_zero_config_still_loads(self):
        for angle in (float('nan'),float('inf')):
            o=self.settings();o['gripper']['closed_angle_deg']=angle
            with patch.object(app,'read_yaml',return_value=o),self.assertRaises(ValueError):app.load_workflow('unused')
        for angle in (0.,10.):
            o=self.settings();o['startup_view']['joints_deg']=[0,0,-10,-20,0,0,angle]
            o.pop('initial_poses',None)
            if angle==0.:o['gripper'].pop('closed_angle_deg')
            else:o['gripper']['closed_angle_deg']=angle
            with patch.object(app,'read_yaml',return_value=o):
                self.assertEqual(app.load_workflow('unused')['gripper']['closed_angle_deg'],angle)

    def test_ocr_initialization_failure_closes_parallel_camera_before_any_robot(self):
        worker=NS(wait_ready=Mock(side_effect=RuntimeError("OCR failure")),close=Mock())
        camera=NS(open=Mock(),warm_up=Mock(),close=Mock())
        with patch.object(app,"PerceptionWorker",return_value=worker),patch("drivers.camera.make_camera",return_value=camera), \
                patch.object(motion,"Robot") as robot,patch("cv2.destroyAllWindows"),patch("cv2.namedWindow"):
            with self.assertRaisesRegex(RuntimeError,"OCR failure"):
                app.run_elevator({},self.settings(),NS(execute=True,stage="contact",device="cpu",floor=None))
        camera.open.assert_called_once();camera.close.assert_called_once()
        robot.assert_not_called();worker.close.assert_called_once()

    def test_camera_start_failure_releases_camera_and_worker_without_robot(self):
        worker=NS(wait_ready=Mock(),close=Mock())
        camera=NS(open=Mock(side_effect=RuntimeError("camera failure")),close=Mock())
        with patch.object(app,"PerceptionWorker",return_value=worker),patch("drivers.camera.make_camera",return_value=camera), \
                patch.object(motion,"Robot") as robot,patch("cv2.destroyAllWindows"):
            with self.assertRaisesRegex(RuntimeError,"camera failure"):
                app.run_elevator({},self.settings(),NS(execute=True,stage="contact",device="cpu",floor=None))
        robot.assert_not_called();worker.close.assert_called_once();camera.close.assert_called_once()

    def test_cancel_during_parallel_model_loading_never_initializes_robot(self):
        worker=NS(wait_ready=Mock(side_effect=lambda **kw:kw["on_wait"]()),close=Mock())
        camera=NS(open=Mock(),warm_up=Mock(),close=Mock(),
                  get_frame=lambda:(np.zeros((80,100,3),np.uint8),np.ones((80,100),np.uint16)))
        with contextlib.ExitStack() as s:
            s.enter_context(patch.object(app,"PerceptionWorker",return_value=worker))
            s.enter_context(patch("drivers.camera.make_camera",return_value=camera))
            robot=s.enter_context(patch.object(motion,"Robot"))
            for name in ("namedWindow","imshow","destroyAllWindows"):s.enter_context(patch("cv2."+name))
            s.enter_context(patch("cv2.waitKey",return_value=ord("q")))
            s.enter_context(patch("cv2.getWindowProperty",return_value=1))
            self.assertEqual(app.run_elevator({},self.settings(),NS(execute=True,stage="contact",device="cpu",floor=None)),0)
        robot.assert_not_called();camera.close.assert_called_once();worker.close.assert_called_once()


class StableObservationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_geometry import snapshot,button
        from dataclasses import replace
        self.snap=snapshot();self.b=button()
        self.result=NS(errors=[],panel_bbox=(5,5,95,75),buttons=[self.b])
        self.current=replace(self.snap,captured_at=1.8,frame_id=2)
        self.tracker=NS(panel=NS(stable=True),remembered_associations=lambda *a,**kw:())
        self.request=dict(at=1.7,floor="3")
        self.settings=dict(stable_snapshot_age_s=2.,max_result_age_s=4.)
        self.cache={}
        app.remember_stable(self.cache,self.snap,self.result,self.tracker,.8)

    def resolve(self,last=None,snapshot=None):
        return app.resolve_observation(self.request,last or self.result,snapshot or self.snap,self.cache,
                                       self.tracker,self.current,self.settings,.8)

    def test_pre_input_stable_result_is_usable_without_another_ocr_result(self):
        snap,b,source=self.resolve()
        self.assertIs(snap,self.snap);self.assertIs(b,self.b);self.assertEqual(source,"STABLE_CACHE")

    def test_fresh_result_preferred_and_disabled_cache_waits_for_fresh(self):
        from dataclasses import replace
        new=replace(self.snap,captured_at=1.75,frame_id=3)
        snap,_,source=self.resolve(snapshot=new)
        self.assertIs(snap,new);self.assertEqual(source,"NEW_RESULT")
        self.settings["stable_snapshot_age_s"]=0.
        with self.assertRaises(ValueError):self.resolve()

    def test_unknown_latest_ocr_can_reuse_stable_bundle_but_conflict_cannot(self):
        from tests.test_geometry import button
        unknown=NS(errors=[],panel_bbox=self.result.panel_bbox,buttons=[button(None,"NO_TEXT",None)])
        self.assertEqual(self.resolve(last=unknown)[2],"STABLE_CACHE")
        for label in ("4","8"):
            conflicting=NS(errors=[],panel_bbox=self.result.panel_bbox,buttons=[button(label,"UNSTABLE",.9)])
            with self.assertRaisesRegex(ValueError,"冲突"):self.resolve(last=conflicting)
        duplicate=button("3");duplicate.ring.center=(85,50)
        with self.assertRaisesRegex(ValueError,"冲突"):
            self.resolve(last=NS(errors=[],panel_bbox=self.result.panel_bbox,buttons=[self.b,duplicate]))

    def test_panel_loss_ocr_error_and_unstable_ocr_do_not_create_reusable_targets(self):
        self.tracker.panel.stable=False
        with self.assertRaises(ValueError):self.resolve()
        app.remember_stable(self.cache,self.snap,self.result,self.tracker,.8)
        self.assertFalse(self.cache)
        self.tracker.panel.stable=True;self.b.status="UNSTABLE"
        app.remember_stable(self.cache,self.snap,self.result,self.tracker,.8)
        self.assertFalse(self.cache)
        self.b.status="STABLE"
        app.remember_stable(self.cache,self.snap,self.result,self.tracker,.8)
        self.result.errors=["OCR_ERROR"]
        with self.assertRaises(ValueError):self.resolve()
        app.remember_stable(self.cache,self.snap,self.result,self.tracker,.8)
        self.assertFalse(self.cache)

class VerifiedPixelBundleChecks(unittest.TestCase):
    def setUp(self):
        from tests.test_geometry import snapshot,button
        from dataclasses import replace
        self.snap=snapshot();self.button=button();self.request=dict(at=1.7,floor='3')
        self.old=NS(errors=[],panel_bbox=(5,5,95,75),buttons=[self.button],tracking=dict(workflow_evidence=True,
                    region_evidence={'0':dict(source='CURRENT_OCR',pixel_match=True,track_id=42)},display_label_cache={}))
        self.current=replace(self.snap,captured_at=1.8,frame_id=2)
        self.tracker=NS(panel=NS(stable=True),remembered_associations=lambda *a,**kw:())
        self.settings=dict(stable_snapshot_age_s=2.,max_result_age_s=4.)
        self.cache={'3':(self.snap,self.old)}

    def latest(self,*,identity=42,source='PIXEL_MATCHED_CACHE',pixels=True):
        from tests.test_geometry import button
        return NS(errors=[],panel_bbox=self.old.panel_bbox,buttons=[button(None,'NO_TEXT',None)],
                  tracking=dict(workflow_evidence=True,region_evidence={'0':dict(source=source,pixel_match=pixels,track_id=identity)},
                                display_label_cache={'0':dict(text='3',confidence=.95)}))

    def resolve(self,last,snapshot=None):
        return app.resolve_observation(self.request,last,snapshot or self.snap,self.cache,self.tracker,
                                       self.current,self.settings,.8)

    def test_same_pixels_and_identity_allow_original_complete_rgbd_bundle(self):
        snap,b,source=self.resolve(self.latest())
        self.assertIs(snap,self.snap);self.assertIs(b,self.button);self.assertEqual(source,'STABLE_CACHE')

    def test_invalidated_or_changed_identity_cannot_fall_back_to_old_bundle(self):
        for kwargs in (dict(identity=43),dict(source='NO_TEXT'),dict(pixels=False)):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'证据已改变'):
                self.resolve(self.latest(**kwargs))

    def test_duplicate_cached_floor_at_distant_rim_rejects_old_bundle(self):
        from tests.test_geometry import button
        last=self.latest();other=button(None,'NO_TEXT',None);other.ring=NS(center=(85,50),radius=12.)
        last.buttons.append(other)
        last.tracking['region_evidence']['1']=dict(source='PIXEL_MATCHED_CACHE',pixel_match=True,track_id=43)
        last.tracking['display_label_cache']['1']=dict(text='3',confidence=.95)
        with self.assertRaisesRegex(ValueError,'冲突'):self.resolve(last)

    def test_proven_label_memory_uses_current_snapshot_depth_and_pose(self):
        from dataclasses import replace
        last=self.latest();new=replace(self.snap,captured_at=1.75,frame_id=3,depth_mm=self.snap.depth_mm+1)
        self.tracker.remembered_associations=lambda *a,**kw:[NS(label='3',observation_index=0)]
        snap,b,source=self.resolve(last,new)
        self.assertIs(snap,new);self.assertIsNone(b.text)
        self.assertEqual(source,'CURRENT_RIM_LABEL_MEMORY')
        point,_=geometry.point_from_button(snap,b)
        self.assertAlmostEqual(point[2],.401)
