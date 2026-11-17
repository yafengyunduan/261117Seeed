"""Offline current-frame localization and base-frame pitch-zero tests."""
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
from elevator import geometry as g
from utils.transforms import pose6d_to_mat4,rotation_matrix_to_euler_zyx

def snapshot():
    return g.Snapshot(1,1.,np.zeros((80,100,3),np.uint8),np.full((80,100),400,np.uint16),
        np.array([[100.,0,50],[0,100.,40],[0,0,1]]),np.zeros(5),np.eye(4))

def button(text="3",status="STABLE",confidence=.9):
    return NS(text=text,status=status,confidence=confidence,text_center_px=(1,1),ring=NS(center=(60,50),radius=12.))

class GeometryTests(unittest.TestCase):
    def test_latched_bundle_has_no_time_expiry_but_preserves_scene_and_timestamp_checks(self):
        old=snapshot();current=replace(old,frame_id=2,captured_at=1000.)
        g.check_cached_observation(old,current,button(),None)
        for changed in (replace(current,captured_at=.5),replace(current,captured_at=float('nan')),
                        replace(current,generation=1),replace(current,color=old.color+100),
                        replace(current,depth_mm=old.depth_mm+10),
                        replace(current,T_base_end=pose6d_to_mat4(.01,0,0,0,0,0))):
            with self.subTest(timestamp=changed.captured_at),self.assertRaises(ValueError):
                g.check_cached_observation(old,changed,button(),None)

    def test_stationary_cached_rgbd_bundle_accepts_small_sensor_noise(self):
        old=snapshot();current=replace(old,frame_id=2,captured_at=1.5,color=old.color+2,depth_mm=old.depth_mm+1)
        g.check_cached_observation(old,current,button())

    def test_cached_bundle_rejects_expiration_generation_pose_image_depth_and_intrinsics_changes(self):
        old=snapshot();current=replace(old,frame_id=2,captured_at=1.5)
        moved=pose6d_to_mat4(.003,0,0,0,0,0)
        rotated=pose6d_to_mat4(0,0,0,0,0,np.radians(.6))
        for changed in (replace(current,captured_at=3.1),replace(current,captured_at=.9),
                        replace(current,generation=1),replace(current,T_base_end=moved),replace(current,T_base_end=rotated),
                        replace(current,color=np.full_like(old.color,80)),replace(current,depth_mm=old.depth_mm+10),
                        replace(current,depth_mm=np.zeros_like(old.depth_mm)),replace(current,K=old.K*1.01)):
            with self.subTest(changed=changed.frame_id),self.assertRaises(ValueError):
                g.check_cached_observation(old,changed,button())

    def test_ring_center_depth_and_base_compensation_match_main_chain(self):
        snap=snapshot();b=button();T=pose6d_to_mat4(.26,0,.192,0,.2,0)
        snap=replace(snap,T_base_end=T)
        camera,base=g.point_from_button(snap,b,np.eye(4),{"calibration":{"hand_eye_compensation_m":{"x":-.02,"y":-.02,"z":-.05}}})
        np.testing.assert_allclose(camera,[.04,.04,.4])
        np.testing.assert_allclose(base,(T@np.r_[camera,1])[:3]+[-.02,-.02,-.05])

    def test_preview_does_not_invent_base_position(self):
        _,base=g.point_from_button(snapshot(),button())
        self.assertIsNone(base)
        self.assertIsNone(g.offset_target_base(base,[.05,.0235,.0139]))

    def test_elevator_correction_is_added_in_base_axes_without_mutating_raw_observation(self):
        snap=replace(snapshot(),T_base_end=pose6d_to_mat4(.26,.01,.192,.1,-.5,.3))
        original=np.array([-.02,-.02,-.05]);delta=np.array([.05,.0235,.0139])
        camera,raw=g.point_from_button(snap,button(),np.eye(4),
            {"calibration":{"hand_eye_compensation_m":dict(zip("xyz",original))}})
        saved=raw.copy();corrected=g.offset_target_base(raw,delta)
        expected=(snap.T_base_end@np.r_[camera,1])[:3]+original+delta
        np.testing.assert_allclose(corrected,expected);np.testing.assert_array_equal(raw,saved)
        level=g.level_pose(snap.T_base_end)
        old=g.button_poses(raw,level,[0,0,0]);new=g.button_poses(corrected,level,[0,0,0])
        for stage in old:
            np.testing.assert_allclose(new[stage][:3,3]-old[stage][:3,3],delta)
        self.assertAlmostEqual(np.linalg.norm(new["surface"][:3,3]-new["approach"][:3,3]),.05)

    def test_invalid_base_corrections_never_generate_targets(self):
        for offset in ([0,0],[0,0,float("nan")],[float("inf"),0,0]):
            with self.assertRaises(ValueError):g.offset_target_base([.4,-.1,.4],offset)

    def test_invalid_depth_or_pixel_never_falls_back_to_a_target(self):
        snap=snapshot()
        for depth in (np.zeros((80,100)),np.full((80,100),np.nan),np.full((80,100),2000)):
            with self.assertRaises(ValueError):g.point_from_button(replace(snap,depth_mm=depth),button())
        b=button();b.ring.center=(-1,50)
        with self.assertRaises(ValueError):g.point_from_button(snap,b)

    def test_pitch_exactly_zero_preserves_yaw_and_tip(self):
        T=pose6d_to_mat4(.26,.01,.192,.12,-.52,.31);offset=np.array([.01,.02,.03])
        level=g.level_pose(T,offset)
        np.testing.assert_allclose(rotation_matrix_to_euler_zyx(level[:3,:3]),[0,0,.31],atol=1e-10)
        np.testing.assert_allclose(level[:3,3]+level[:3,:3]@offset,T[:3,3]+T[:3,:3]@offset)

    def test_button_poses_reach_surface_then_retreat_horizontally(self):
        level=g.level_pose(pose6d_to_mat4(.26,0,.192,0,-.52,.3))
        offset=np.array([.02,0,0]);point=np.array([.4,-.1,.4])
        poses=g.button_poses(point,level,offset,.05,.003)
        np.testing.assert_allclose(poses["surface"][:3,3]+poses["surface"][:3,:3]@offset,point)
        np.testing.assert_allclose(poses["approach"],poses["retreat"])
        delta=poses["press"][:3,3]-poses["surface"][:3,3]
        self.assertAlmostEqual(np.linalg.norm(delta),.003)
        self.assertAlmostEqual(delta[2],0.)
        for T in poses.values():self.assertAlmostEqual(rotation_matrix_to_euler_zyx(T[:3,:3])[1],0.)

    def test_floor_selection_and_memory_require_unique_current_rim(self):
        b=button();result=NS(buttons=[b,button("HLD")])
        self.assertIs(g.select_floor(result,None,"３"),b)
        result.buttons.append(button())
        with self.assertRaises(ValueError):g.select_floor(result,None,"3")
        result.buttons=[button(None,"NO_TEXT",None)]
        tracker=NS(remembered_associations=lambda r,now: [NS(label="3",observation_index=0)])
        self.assertIs(g.select_floor(result,tracker,"3",now=2),result.buttons[0])
        with self.assertRaises(ValueError):g.select_floor(result,None,"3")
        result.buttons.append(button(status="UNSTABLE"))
        with self.assertRaisesRegex(ValueError,"冲突"):g.select_floor(result,tracker,"3",now=2)

    def test_free_orientation_keeps_horizontal_approach_and_correct_tool_tip(self):
        current=pose6d_to_mat4(.26,0,.35,.2,-.7,.3)
        offset=np.array([.02,.01,.03]);point=np.array([.47,-.03,.734])
        direction=g.horizontal_approach_direction(current)
        poses=g.button_poses(point,current,offset,.1,.003,forward_base=direction)
        np.testing.assert_allclose(direction,[np.cos(.3),np.sin(.3),0.])
        for name,amount in (('approach',-.1),('surface',0.),('press',.003),('retreat',-.1)):
            T=poses[name]
            np.testing.assert_allclose(T[:3,:3],current[:3,:3])
            np.testing.assert_allclose(T[:3,3]+T[:3,:3]@offset,point+amount*direction)

    def test_invalid_explicit_approach_direction_is_rejected(self):
        for direction in ([0,0,0],[1,2],[float('nan'),0,1]):
            with self.assertRaises(ValueError):g.button_poses([.4,0,.4],np.eye(4),[0,0,0],forward_base=direction)

class WorkflowDisplayCacheSelectionTests(unittest.TestCase):
    def result(self):
        b=button(None,'NO_TEXT',None)
        return NS(errors=[],buttons=[b],tracking=dict(workflow_evidence=True,
            region_evidence={'0':dict(source='PIXEL_MATCHED_CACHE',pixel_match=True,track_id=1)},
            display_label_cache={'0':dict(text='3',confidence=.95)}))

    def test_display_cache_alone_is_not_a_stable_target(self):
        result=self.result()
        with self.assertRaises(ValueError):g.select_floor(result,None,'3')
        tracker=NS(remembered_associations=lambda r,now:[NS(label='3',observation_index=0)])
        self.assertIs(g.select_floor(result,tracker,'3'),result.buttons[0])

    def test_cached_duplicate_digit_at_another_rim_remains_ambiguous(self):
        result=self.result();other=button(None,'NO_TEXT',None);other.ring=NS(center=(80,50),radius=12.)
        result.buttons.append(other)
        result.tracking['region_evidence']['1']=dict(source='PIXEL_MATCHED_CACHE',pixel_match=True,track_id=2)
        result.tracking['display_label_cache']['1']=dict(text='3',confidence=.95)
        tracker=NS(remembered_associations=lambda r,now:[NS(label='3',observation_index=0)])
        with self.assertRaisesRegex(ValueError,'多个圆环'):g.select_floor(result,tracker,'3')
