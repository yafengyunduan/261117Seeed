"""New CLI checks: no devices or packages initialized in simulation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from elevator import app

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('journey_entry',ROOT/'scripts/261113_my_elevator_journey.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class JourneyEntryTests(unittest.TestCase):
    def test_shared_file_controls_all_four_poses_and_simulation_without_rewriting_default_config(self):
        options=app.read_yaml(ROOT/'config/261105_my_elevator.yaml')
        initial=dict(grasp_ready=dict(x=.27,y=.01,z=.31,roll=0.,pitch=.6,yaw=.1,duration=2.5),
                     call_view_deg=[0,0,-20,-20,0,0],floor_view_deg=[0,0,-30,-20,0,0],
                     display_view_deg=[0,0,-45,-20,0,0])
        options['initial_poses']=initial
        original=(ROOT/'config/default.yaml').read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'workflow.yaml';path.write_text(yaml.safe_dump(options))
            argv=['--workflow-config',str(path)]
            with patch.object(entry,'Journey') as factory:
                factory.return_value.run.return_value=0
                self.assertEqual(entry.main(argv),0)
            cfg,parsed,settings,_=factory.call_args.args
            self.assertEqual(cfg['robot']['ready_pose'],initial['grasp_ready'])
            self.assertEqual(parsed['startup_view']['joints_deg'],initial['floor_view_deg'])
            self.assertEqual(settings['display_view_deg'],initial['display_view_deg'])
            with patch.object(entry,'Journey',side_effect=AssertionError('device')),contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(entry.main([*argv,'--simulate']),0)
            plan=json.loads(output.getvalue())
            self.assertEqual(plan['grasp_ready_pose'],initial['grasp_ready'])
            self.assertEqual(plan['view_joints_deg'],{p:initial[k] for p,k in
                (('CALL','call_view_deg'),('FLOOR','floor_view_deg'),('DISPLAY','display_view_deg'))})
            self.assertFalse(plan['devices_created'])
        self.assertEqual((ROOT/'config/default.yaml').read_bytes(),original)

    def test_yaml_placement_and_cli_override_are_used_by_runtime_and_simulation(self):
        options=app.read_yaml(ROOT/'config/261105_my_elevator.yaml')
        options['grasp_place']={'xyz_m':[.28,-.32,.36]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'workflow.yaml';path.write_text(yaml.safe_dump(options))
            original=path.read_bytes()
            for extra,expected in (([],[.28,-.32,.36]),
                                   (['--place-xyz-m','.3','-.25','.4'],[.3,-.25,.4])):
                argv=['--workflow-config',str(path),*extra]
                with self.subTest(extra=extra),patch.object(entry,'Journey') as factory:
                    factory.return_value.run.return_value=0
                    self.assertEqual(entry.main(argv),0)
                    self.assertEqual(factory.call_args.args[3].place_xyz_m,expected)
                with patch.object(entry,'Journey',side_effect=AssertionError('device')),contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(entry.main([*argv,'--simulate']),0)
                pose=json.loads(output.getvalue())['place_pose']
                self.assertEqual([row[3] for row in pose[:3]],expected)
            self.assertEqual(path.read_bytes(),original)

    def test_default_preview_keeps_user_threshold_and_existing_compensation(self):
        with patch.object(entry,'Journey') as factory:
            factory.return_value.run.return_value=0
            self.assertEqual(entry.main([]),0)
        cfg,options,settings,args=factory.call_args.args
        original=app.read_yaml(ROOT/'config/261105_my_elevator.yaml')
        self.assertFalse(args.execute);self.assertEqual(args.press_stage,'contact')
        self.assertEqual(options['elevator']['panel_type'],'physical')
        self.assertEqual(options['elevator']['press_travel_m'],original['elevator']['press_travel_m'])
        self.assertEqual(options['ocr']['min_confidence'],original['ocr']['min_confidence'])
        self.assertEqual(options['elevator']['target_offset_base_m'],original['elevator']['target_offset_base_m'])
        self.assertIsNone(settings['call_target_offset_base_m'])
        self.assertEqual(entry.effective_call_offset(settings,options).tolist(),original['elevator']['target_offset_base_m'])
        self.assertEqual(settings['search_timeout_s'],10.)
        self.assertEqual(args.floor_input_delay_s,.4)
        self.assertEqual(args.button_orientation,'free')

    def test_simulation_never_constructs_runtime(self):
        with patch.object(entry,'Journey',side_effect=AssertionError('device')),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(entry.main(['--simulate']),0)
        result=json.loads(stream.getvalue())
        self.assertFalse(result['devices_created'])
        self.assertEqual(result['button_orientation'],'free');self.assertTrue(result['position_only_ik'])
        self.assertEqual(result['stages'][1:4],['七轴抬头闭合','等待C开始外呼识别','稳定两键U/D'])
        self.assertEqual(result['search_failure_traces']['CALL'],[-15.,-15.,'HOME'])
        self.assertEqual(result['search_failure_traces']['DISPLAY'],[-15.,-10.,'HOME'])
        self.assertEqual(result['call_target_offset_base_m'],result['floor_target_offset_base_m'])
        self.assertEqual(result['floor_input_delay_s'],.4)
        self.assertEqual(result['skip_key'],'N')
        self.assertEqual(set(result['skip_transitions']),{'WAIT_CALL','CALL','FLOOR','DISPLAY'})
        self.assertIn('home',result['skip_transitions']['DISPLAY'])
        self.assertIsNone(result['floor_cache_age_s'])
        self.assertTrue(result['floor_cache_until_update'])

    def test_invalid_coordinates_or_duration_fail_before_runtime(self):
        for args in (['--place-xyz-m','nan','0','0'],['--place-duration-s','0'],['--release-wait-s','-1'],['--standoff-m','.3'],
                     ['--floor-input-delay-s','nan'],['--floor-input-delay-s','-1'],['--floor-input-delay-s','2.1']):
            with self.subTest(args=args),patch.object(entry,'Journey',side_effect=AssertionError('device')),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(entry.main(args),2)

    def test_contact_is_explicit_and_only_overrides_in_memory(self):
        with patch.object(entry,'Journey') as factory:
            factory.return_value.run.return_value=0
            self.assertEqual(entry.main(['--execute','--press-stage','contact','--ocr-device','cpu']),0)
        _,options,_,args=factory.call_args.args
        self.assertTrue(args.execute);self.assertEqual(args.press_stage,'contact')
        self.assertEqual(options['ocr']['device'],'cpu')

    def test_approach_remains_an_explicit_diagnostic_override(self):
        with patch.object(entry,'Journey') as factory:
            factory.return_value.run.return_value=0
            self.assertEqual(entry.main(['--execute','--press-stage','approach']),0)
        self.assertEqual(factory.call_args.args[3].press_stage,'approach')

    def test_floor_input_pause_can_be_adjusted_without_changing_config(self):
        with patch.object(entry,'Journey') as factory:
            factory.return_value.run.return_value=0
            self.assertEqual(entry.main(['--floor-input-delay-s','.7']),0)
        self.assertEqual(factory.call_args.args[3].floor_input_delay_s,.7)

    def test_planning_failure_is_still_reported_as_nonzero_exit(self):
        import numpy as np
        from elevator.motion import TrajectoryPlanningError
        with patch.object(entry,'Journey') as factory,contextlib.redirect_stderr(io.StringIO()) as stream:
            factory.return_value.run.side_effect=TrajectoryPlanningError(np.eye(4),3.)
            self.assertEqual(entry.main([]),2)
        self.assertIn('TrajectoryPlanningError',stream.getvalue())

    def test_level_orientation_is_an_explicit_legacy_option(self):
        with patch.object(entry,'Journey') as factory:
            factory.return_value.run.return_value=0
            self.assertEqual(entry.main(['--button-orientation','level']),0)
        self.assertEqual(factory.call_args.args[3].button_orientation,'level')


if __name__=='__main__':unittest.main()
