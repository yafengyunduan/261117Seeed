"""Independent experiment: pixel/layout checks, hardware and model access mocked."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'scripts/261111_my_panel_completion.py'
spec=importlib.util.spec_from_file_location('panel_completion',ENTRY)
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
v=experiment.vision
from elevator import panel_261112 as recognition
from elevator import geometry


def buttons_for(floors):
    return [v.ButtonObservation(v.Ring(100+80*(i%2),1000-80*(i//2),25,1.),str(n),.99)
            for i,n in enumerate(floors)]


def hide(buttons,indices):
    for i in indices:buttons[i].text=None;buttons[i].confidence=None


def door_image(kind='OPEN',invert=False):
    image=np.full((140,140,3),70,np.uint8)
    if kind=='OPEN':
        triangles=[[(36,70),(58,51),(58,89)],[(104,70),(82,51),(82,89)]]
    else:
        triangles=[[(58,70),(36,51),(36,89)],[(82,70),(104,51),(104,89)]]
    for points in triangles:cv2.fillPoly(image,[np.array(points,np.int32)],(255,255,255))
    cv2.rectangle(image,(68,51),(72,89),(255,255,255),-1)
    return 255-image if invert else image


class CompletionLayoutTests(unittest.TestCase):
    def test_unknown_bottom_keeps_zero_and_negative_alternatives(self):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8]);hide(buttons,[0,1])
        r=experiment.layout_candidates(buttons,{},{});
        self.assertEqual(r['status'],'CANDIDATES_ONLY')
        self.assertEqual(set(r['buttons'][0]['preferred_candidates']),{'-1','0'})
        self.assertIn('0',r['buttons'][0]['floor_candidates'])
        self.assertIn('-3',r['buttons'][0]['floor_candidates'])
        self.assertTrue(all(not b['executable'] for b in r['buttons']))
        self.assertIsNone(buttons[0].text)

    def test_visible_minus_and_vertical_support_candidates_without_overwriting_ocr(self):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8]);hide(buttons,[0,1])
        glyphs={0:{'shape_candidates':['-1']},1:{'shape_candidates':['1']}}
        r=experiment.layout_candidates(buttons,{},glyphs)
        for i,label in ((0,'-1'),(1,'1')):
            self.assertEqual(r['buttons'][i]['shape_layout_agreement'],[label])
            self.assertEqual(r['buttons'][i]['status'],'SHAPE_LAYOUT_CANDIDATE')
            self.assertIsNone(r['buttons'][i]['probability']);self.assertIsNone(buttons[i].text)
        query=experiment.target_candidates({'layout':r},-1)
        self.assertEqual(query[0]['index'],0)
        self.assertEqual(query[0]['requested_floor_support'],'SHAPE_LAYOUT_SUPPORT')
        self.assertTrue(all(not c['executable'] for c in query))

    def test_unread_top_cannot_rule_out_9_to_12_or_missing_rims(self):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8]);hide(buttons,[8])
        r=experiment.layout_candidates(buttons,{},{});
        self.assertEqual(r['buttons'][8]['preferred_candidates'],['8'])
        for label in ('8','9','10','11','12'):self.assertIn(label,r['buttons'][8]['floor_candidates'])
        self.assertTrue(r['buttons'][8]['requires_confirmation'])

    def test_shape_can_support_unread_8_while_skipped_alternatives_remain_auditable(self):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8]);hide(buttons,[8])
        r=experiment.layout_candidates(buttons,{}, {8:{'shape_candidates':['8']}})
        self.assertEqual(r['buttons'][8]['shape_layout_agreement'],['8'])
        self.assertIn('12',r['buttons'][8]['layout_possible_floors'])
        self.assertFalse(r['buttons'][8]['executable'])

    def test_full_inventory_generalizes_negatives_and_two_digit_floors(self):
        floors=[-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12]
        buttons=buttons_for(floors);hidden=[0,1,12,13,14];hide(buttons,hidden)
        r=experiment.layout_candidates(buttons,{}, {},floor_list=floors,inventory_complete=True)
        for i in hidden:self.assertEqual(r['buttons'][i]['floor_candidates'],[str(floors[i])])
        self.assertEqual(r['grid']['row_count'],8)

    def test_allowed_inventory_does_not_imply_every_floor_is_present(self):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8]);hide(buttons,[8])
        r=experiment.layout_candidates(buttons,{}, {},floor_list=[-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12])
        self.assertEqual(r['buttons'][8]['floor_candidates'],['8','9','10','11','12'])

    def test_multiple_scan_orders_are_retained(self):
        buttons=[]
        for i in range(9):
            buttons.append(v.ButtonObservation(v.Ring(100+80*(i%3),400-80*(i//3),25,1.),
                            str(i+1) if i in (0,4,8) else None,.99 if i in (0,4,8) else None))
        r=experiment.layout_candidates(buttons,{}, {})
        self.assertGreater(len(r['models']),1)
        self.assertIn('2',r['buttons'][1]['floor_candidates'])
        self.assertIn('4',r['buttons'][1]['floor_candidates'])

    def test_nearby_door_and_hold_never_become_negative_floors(self):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8]);hide(buttons,[0])
        buttons.extend([v.ButtonObservation(v.Ring(100+80*i,1120,25,1.),variants=[('HOLD',.99)]) for i in range(3)])
        doors={9:{'label':'OPEN'},10:{'label':'CLOSE'}}
        r=experiment.layout_candidates(buttons,doors,{})
        for row in r['buttons'][9:]:
            self.assertEqual(row['status'],'FUNCTION_OR_WORD');self.assertEqual(row['floor_candidates'],[])

    def test_duplicate_labels_insufficient_anchors_and_shape_conflict_are_explicit(self):
        self.assertEqual(experiment.layout_candidates([],{}, {})['status'],'INSUFFICIENT_ANCHORS')
        self.assertEqual(experiment.layout_candidates(buttons_for([1,1,3]),{}, {})['status'],'DUPLICATE_ANCHOR_LABELS')
        buttons=buttons_for([1,2,3,4,5,6]);hide(buttons,[3])
        r=experiment.layout_candidates(buttons,{}, {3:{'shape_candidates':['-1']}})
        self.assertEqual(r['buttons'][3]['status'],'SHAPE_LAYOUT_CONFLICT')
        self.assertEqual(r['buttons'][3]['floor_candidates'],['4'])

    def test_floor_parsing_does_not_map_letters_to_digits(self):
        for text,value in (('-3',-3),('-2',-2),('9',9),('10',10),('11',11),('12',12),('−1',-1)):
            self.assertEqual(experiment.floor_value(text),value)
        for text in ('H','Y','I','B','HOLD','111','3abc'):
            self.assertIsNone(experiment.floor_value(text))

    def test_parser_has_no_execute_flag_and_rejects_invalid_inventory(self):
        for args in (['--execute'],['--inventory-complete'],['--floor-list=1,1'],['--floor-list=100'],['--ocr-min-score','nan']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):experiment.parse_args(args)

    def test_domain_ends_reserve_capacity_for_other_slots(self):
        buttons=[v.ButtonObservation(v.Ring(100,100+60*i,20,1.),str(i+1) if i in (2,3,4) else None,
                                    .99 if i in (2,3,4) else None) for i in range(6)]
        r=experiment.layout_candidates(buttons,{}, {},floor_list=[1,2,3,4,5,6])
        self.assertEqual(r['buttons'][1]['floor_candidates'],['2'])

    def test_entire_unread_column_is_measured_and_not_silently_discarded(self):
        buttons=buttons_for([1,2,3,4,5,6]);hide(buttons,[1,3,5])
        r=experiment.layout_candidates(buttons,{}, {},floor_list=[1,2,3,4,5,6],inventory_complete=True)
        self.assertEqual(r['status'],'CANDIDATES_ONLY')
        self.assertEqual(len(r['grid']['indices']),6)
        for i in (1,3,5):self.assertEqual(r['buttons'][i]['floor_candidates'],[str(i+1)])

    def test_geometric_filter_cannot_ignore_a_conflicting_numeric_anchor(self):
        buttons=[v.ButtonObservation(v.Ring(x,100+30*i,10,1.),str(n),.99)
                 for i,(x,n) in enumerate(zip((100,108,112,114),(99,2,3,4)))]
        r=experiment.layout_candidates(buttons,{}, {},floor_list=[1,2,3,4,99])
        self.assertEqual(r['status'],'IRREGULAR_OR_UNSUPPORTED_GRID')

    def test_shape_preference_does_not_remove_other_possible_target_queries(self):
        buttons=buttons_for([1,2,3,4]);hide(buttons,[3])
        r=experiment.layout_candidates(buttons,{}, {3:{'shape_candidates':['8']}},floor_list=list(range(1,10)))
        self.assertEqual(r['buttons'][3]['shape_layout_agreement'],['8'])
        self.assertIn('4',r['buttons'][3]['floor_candidates'])
        self.assertEqual(len(experiment.target_candidates({'layout':r},4)),1)

    def test_conflicting_ocr_numbers_stay_explicit_despite_shape_and_layout(self):
        buttons=buttons_for([1,2,3,4]);hide(buttons,[0])
        buttons[0].variants=[('1',.97),('-1',.95),('-1',.96)]
        r=experiment.layout_candidates(buttons,{}, {0:{'shape_candidates':['1']}},
                                       floor_list=[1,2,3,4],inventory_complete=True)
        self.assertEqual(r['buttons'][0]['status'],'OCR_NUMERIC_CONFLICT')
        self.assertEqual(r['buttons'][0]['ocr_numeric_alternatives'],['-1','1'])


class RegionMemoryTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.rings=[v.Ring(90+100*i,100,36,1.) for i in range(4)]
        self.image=np.full((220,500,3),80,np.uint8)
        for i,r in enumerate(self.rings):
            cv2.circle(self.image,r.center,36,(220,220,220),2)
            cv2.putText(self.image,str(i+4),(int(r.x-13),110),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
        self.engine=Mock(backend_info={},min_score=.7,last_inference_stats={})
        self.engine.recognize.side_effect=lambda image,rings,box:[
            v.ButtonObservation(r,str(round((r.x-90)/100)+4),.95,
                                (round(r.x-10),90,round(r.x+10),112),[(str(round((r.x-90)/100)+4),.95)]) for r in rings]
        self.ocr=experiment.RegionOCR(self.engine,budget=2)
        self.ocr.ready=True

    def run_frame(self,at,image=None,rings=None):
        self.ocr.now=at
        return self.ocr.recognize(self.image if image is None else image,
                                  self.rings if rings is None else rings,(0,0,500,220))

    def stabilize(self):
        for at in (1.,1.2,1.4,1.6,1.8,2.):self.run_frame(at)

    def test_budget_confirmations_and_cache_are_not_fresh_ocr(self):
        first=self.run_frame(1.)
        self.assertEqual(sum(b.text is not None for b in first),2)
        self.assertTrue(all(b.status!='STABLE' for b in first))
        second=self.run_frame(1.2)
        self.assertTrue(all(b.status!='STABLE' for b in second))
        for at in (1.4,1.6,1.8,2.):last=self.run_frame(at)
        self.assertEqual([b.text for b in last],['4','5','6','7'])
        self.assertTrue(all(b.observations==3 for b in last))
        calls=self.engine.recognize.call_count
        cached=self.run_frame(2.1)
        self.assertEqual(calls,self.engine.recognize.call_count)
        self.assertTrue(all(b.status=='STABLE_CACHE' for b in cached))
        self.assertTrue(all(e['source']=='PIXEL_MATCHED_CACHE' for e in self.ocr.evidence.values()))
        self.assertTrue(all(len(call.args[1])<=2 for call in self.engine.recognize.call_args_list))

    def test_failed_refresh_keeps_stable_4_5_but_expiry_removes_them(self):
        self.stabilize()
        self.engine.recognize.side_effect=lambda image,rings,box:[v.ButtonObservation(r) for r in rings]
        refreshed=self.run_frame(4.1)
        self.assertEqual([b.text for b in refreshed[:2]],['4','5'])
        self.assertTrue(all(e['source']=='PIXEL_MATCHED_CACHE' for e in self.ocr.evidence.values()))
        self.ocr.ready=False
        self.assertTrue(all(b.text is None for b in self.run_frame(11.)))

    def test_changed_digit_occlusion_or_blank_invalidates_old_cache_immediately(self):
        self.stabilize();self.ocr.ready=False
        changed=self.image.copy();changed[62:126,62:118]=80
        cv2.putText(changed,'5',(77,110),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),2)
        result=self.run_frame(2.1,changed)
        self.assertIsNone(result[0].text)
        self.assertEqual(result[1].text,'5')
        self.assertEqual(self.ocr.evidence['0']['source'],'NO_TEXT')
        blank=self.image.copy();blank[60:135,150:230]=80
        self.assertIsNone(self.run_frame(2.2,blank)[1].text)

    def test_numeric_conflict_clears_cache_instead_of_hiding_behind_old_label(self):
        self.stabilize()
        self.engine.recognize.side_effect=lambda image,rings,box:[
            v.ButtonObservation(r,variants=[('4',.97),('5',.96)]) for r in rings]
        result=self.run_frame(4.1)
        self.assertTrue(all(b.text is None for b in result[:2]))
        self.assertEqual(result[0].variants,[('4',.97),('5',.96)])

    def test_loss_reorder_and_small_translation_preserve_only_matching_current_rims(self):
        self.stabilize();self.ocr.ready=False
        moved=cv2.warpAffine(self.image,np.float32([[1,0,2],[0,1,1]]),(500,220))
        rings=[v.Ring(r.x+2,r.y+1,r.radius,1.) for r in reversed(self.rings[1:])]
        result=self.run_frame(2.1,moved,rings)
        self.assertEqual([b.text for b in result],['7','6','5'])
        self.assertEqual(len(self.ocr.memories),3)
        self.assertEqual(result[-1].text_bbox,(182,91,202,113))

    def test_closure_icons_skip_ocr_even_if_engine_would_read_a_digit(self):
        image=door_image('CLOSE');ring=v.Ring(70,70,45,1.)
        self.ocr.now=1.
        result=self.ocr.recognize(image,[ring],(0,0,140,140))
        self.engine.recognize.assert_not_called()
        self.assertIsNone(result[0].text)
        self.assertEqual(self.ocr.last_doors[0]['label'],'CLOSE')

    def test_icon_replaced_by_digit_is_reclassified_without_old_symbol_memory(self):
        ring=v.Ring(70,70,45,1.);image=door_image('CLOSE')
        self.ocr.now=1.;self.ocr.recognize(image,[ring],(0,0,140,140))
        changed=np.full_like(image,70)
        cv2.putText(changed,'4',(55,85),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3)
        self.ocr.now=1.2;result=self.ocr.recognize(changed,[ring],(0,0,140,140))
        self.engine.recognize.assert_called_once()
        self.assertEqual(result[0].text,'4')
        self.assertEqual(self.ocr.last_doors[0]['label'],'UNKNOWN')

    def test_geometry_not_settled_has_no_ocr_and_offline_bypasses_budget(self):
        self.ocr.ready=False
        self.assertTrue(all(b.text is None for b in self.run_frame(1.)))
        self.engine.recognize.assert_not_called()
        self.ocr.offline=True
        result=self.run_frame(1.2)
        self.assertEqual(len(self.engine.recognize.call_args.args[1]),4)
        self.assertTrue(all(b.status=='SINGLE_SHOT' for b in result))

    def test_unchanged_unread_patches_back_off_after_two_failures(self):
        self.engine.recognize.side_effect=lambda image,rings,box:[v.ButtonObservation(r) for r in rings]
        self.run_frame(1.,rings=self.rings[:1]);self.run_frame(1.2,rings=self.rings[:1])
        self.assertEqual(self.engine.recognize.call_count,2)
        self.run_frame(1.5,rings=self.rings[:1]);self.run_frame(4.9,rings=self.rings[:1])
        self.assertEqual(self.engine.recognize.call_count,2)
        self.run_frame(5.3,rings=self.rings[:1])
        self.assertEqual(self.engine.recognize.call_count,3)

    def test_roi_offset_matches_global_tracks_and_maps_text_boxes(self):
        self.stabilize();self.ocr.ready=False;self.ocr.origin=(45,35)
        rings=[v.Ring(r.x-45,r.y-35,r.radius,1.) for r in self.rings]
        result=self.run_frame(2.1,self.image[35:,45:],rings)
        self.assertEqual([b.text for b in result],['4','5','6','7'])
        self.assertEqual(result[0].text_bbox,(35,55,55,77))

    def test_unknown_function_rows_do_not_get_layout_digit_targets(self):
        buttons=buttons_for([1,2,3,4,5,6,7,8,9,10]);hide(buttons,[8,9])
        image=np.full((1100,300,3),80,np.uint8)
        result=v.FrameResult(1.,image.shape[:2],(50,600,230,1050),'GEOMETRY',buttons,
                             tracking={'door_audit':{},'glyph_evidence':{}})
        record=experiment.analyze(image,result)
        for b in record['layout']['buttons'][8:]:
            self.assertEqual(b['status'],'FUNCTION_OR_UNKNOWN')
            self.assertEqual(b['floor_candidates'],[])
        self.assertEqual(experiment.target_candidates(record,9),[])
        with patch.object(cv2,'putText',wraps=cv2.putText) as put:
            experiment.draw(image,record)
        labels=[call.args[1] for call in put.call_args_list]
        self.assertIn('UNREAD',labels)
        self.assertFalse(any('preferred' in text for text in labels))

    def test_acquisition_then_roi_and_reset_maps_full_image_coordinates(self):
        args=experiment.parse_args([])
        image=np.zeros((600,800,3),np.uint8)
        rings=[v.Ring(x,y,30,1.) for y in (230,330) for x in (360,460)]
        with patch.object(v,'ButtonOCR',return_value=self.engine):
            detector=experiment.FastPanelPerception({},args)
        bounds=v.group_box(rings,image.shape)
        def track(work):
            local=rings if work.shape==image.shape else [v.Ring(r.x-bounds[0],r.y-bounds[1],r.radius,1.) for r in rings]
            return local,v.group_box(local,work.shape),{'geometry_mode':'CURRENT_TRACK'}
        detector.layout.track=Mock(side_effect=track)
        detector.model=object()
        with patch.object(v,'detect_objects',return_value=(None,[])) as yolo:
            first=detector.process(image,captured_at=1.)
            second=detector.process(image,captured_at=1.2)
            self.assertEqual(first.image_shape,image.shape[:2])
            self.assertEqual([b.ring.center for b in second.buttons],[r.center for r in rings])
            self.assertLess(second.tracking['processed_area_ratio'],.15)
            yolo.assert_called_once()
            detector.request_refresh();detector.process(image,captured_at=1.4)
            self.assertEqual(yolo.call_count,2)
        self.engine.recognize.assert_not_called()

    def test_entering_roi_reuses_measured_rims_without_a_second_hough_pass(self):
        args=experiment.parse_args([])
        with patch.object(v,'ButtonOCR',return_value=self.engine):
            detector=experiment.FastPanelPerception({},args)
        image=np.full((500,600,3),60,np.uint8)
        rings=[v.Ring(x,y,32,1.) for y in (180,300) for x in (220,340)]
        for r in rings:cv2.circle(image,r.center,32,(230,230,230),3)
        with patch.object(v,'find_rings',return_value=rings),patch.object(v,'repeated_button_group',side_effect=lambda x:x):
            detector.process(image,captured_at=1.)
            detector.process(image,captured_at=1.2)
        self.assertEqual(detector.hough_calls,1)
        self.assertIsNotNone(detector.roi_bounds)


class PositionSupportTests(unittest.TestCase):
    def frame(self,at=1.,*,label='1',shape=True):
        buttons=buttons_for([-1,1,2,3,4,5,6,7,8])
        for b in buttons:b.observations=3;b.status='STABLE'
        hide(buttons,[1]);buttons[1].variants=[('I',.95)]
        result=v.FrameResult(at,(1100,300),(65,620,220,1040),'GEOMETRY',buttons)
        result.tracking.update(geometry_settled_frames=3,
            glyph_evidence={1:{'shape_candidates':[label] if shape else []}},
            region_evidence={str(i):dict(track_id=i+1,pixel_match=True,source='CURRENT_OCR') for i in range(9)},
            door_audit={})
        return result

    def test_one_requires_three_observations_and_preserves_raw_ocr(self):
        support=recognition.PositionSupport()
        for at in (1.,1.2,1.4):
            result=self.frame(at)
            result.tracking['position_inference']=support.update(result)
            self.assertIsNone(result.buttons[1].text)
            self.assertIsNone(result.buttons[1].confidence)
            if at<1.4:
                with self.assertRaises(ValueError):geometry.select_floor(result,None,'1')
        proof=result.tracking['position_inference']['1']
        self.assertEqual(proof['status'],'STABLE_POSITION')
        self.assertIsNone(proof['probability'])
        self.assertIs(geometry.select_floor(result,None,'1'),result.buttons[1])

    def test_no_shape_negative_sign_conflict_duplicate_or_insufficient_anchors_do_not_infer(self):
        for kind in ('shape','minus','conflict','duplicate','few','door','off_grid','two_ones'):
            with self.subTest(kind=kind):
                result=self.frame()
                if kind=='shape':result.tracking['glyph_evidence'][1]['shape_candidates']=[]
                elif kind=='minus':result.tracking['glyph_evidence'][1]['shape_candidates']=['-1']
                elif kind=='conflict':result.buttons[1].variants=[('2',.98)]
                elif kind=='duplicate':result.buttons[3].text='1'
                elif kind=='few':
                    for b in result.buttons[4:]:b.observations=1
                    result.buttons[0].observations=1
                elif kind=='door':result.tracking['door_audit'][1]={'label':'CLOSE'}
                elif kind=='off_grid':result.buttons[1].ring=v.Ring(290.,1000.,25.,1.)
                elif kind=='two_ones':
                    hide(result.buttons,[0]);result.tracking['glyph_evidence'][0]={'shape_candidates':['1']}
                self.assertEqual(recognition.PositionSupport().update(result),{})

    def test_pixel_identity_capture_gap_and_reset_require_new_consensus(self):
        support=recognition.PositionSupport()
        for at in (1.,1.2,1.4):support.update(self.frame(at))
        changed=self.frame(1.6)
        changed.tracking['region_evidence']['1'].update(track_id=88,pixel_match=False)
        self.assertEqual(support.update(changed)['1']['observations'],1)
        self.assertEqual(support.update(self.frame(3.))['1']['observations'],1)
        self.assertEqual(support.update(self.frame(3.))['1']['observations'],1)
        support.reset()
        self.assertEqual(support.update(self.frame(3.2))['1']['observations'],1)

    def test_position_proof_is_bound_to_current_rim_and_raw_digit_still_has_priority(self):
        support=recognition.PositionSupport()
        for at in (1.,1.2,1.4):
            result=self.frame(at);result.tracking['position_inference']=support.update(result)
        self.assertEqual(geometry.position_floor_indices(result,'1'),{1})
        result.buttons[1].ring=v.Ring(190.,1000.,25.,1.)
        self.assertEqual(geometry.position_floor_indices(result,'1'),set())
        result=self.frame(1.6);result.tracking['position_inference']=support.update(result)
        result.buttons[1].text='2';result.buttons[1].confidence=.99
        self.assertEqual(geometry.position_floor_indices(result,'1'),set())

    def test_independent_photo_does_not_authorize_a_position_target(self):
        result=self.frame()
        result.tracking['position_inference']=recognition.PositionSupport().update(result,offline=True)
        self.assertEqual(result.tracking['position_inference']['1']['status'],'POSITION_CANDIDATE')
        self.assertEqual(geometry.position_floor_indices(result,'1'),set())

    def test_unchanged_layout_fits_once_but_drift_or_anchor_change_refits(self):
        support=recognition.PositionSupport()
        with patch.object(recognition,'layout_candidates',wraps=recognition.layout_candidates) as fit:
            for at in (1.,1.2,1.4):support.update(self.frame(at))
            self.assertEqual(fit.call_count,1)
            moved=self.frame(1.6)
            moved.buttons[1].ring=v.Ring(184.,1000.,25.,1.)
            support.update(moved)
            self.assertEqual(fit.call_count,2)
            changed=self.frame(1.8)
            changed.tracking['region_evidence']['2']['track_id']=99
            support.update(changed)
            self.assertEqual(fit.call_count,3)

    def test_selected_position_uses_existing_rgbd_resolver_with_explicit_source(self):
        from elevator import app
        support=recognition.PositionSupport()
        for at in (1.,1.2,1.4):
            result=self.frame(at);result.tracking['position_inference']=support.update(result)
        pose=np.eye(4)
        snap=geometry.Snapshot(7,1.4,np.zeros((1100,300,3),np.uint8),np.full((1100,300),400,np.uint16),
            np.array([[100.,0.,150.],[0.,100.,550.],[0.,0.,1.]]),np.zeros(5),pose)
        tracker=NS(panel=NS(stable=True),remembered_associations=lambda *a,**kw:())
        selected,b,source=app.resolve_observation({'floor':'1','at':1.3},result,snap,{},tracker,snap,
            dict(max_result_age_s=4.,stable_snapshot_age_s=2.),.7)
        self.assertIs(selected,snap);self.assertIs(b,result.buttons[1])
        self.assertEqual(source,'CURRENT_RIM_POSITION_INFERENCE')
        self.assertIsNone(b.text)
        camera,_=geometry.point_from_button(selected,b)
        np.testing.assert_allclose(camera,[.12,1.8,.4])


class CompletionPixelTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.ring=v.Ring(70,70,45,1.)

    def test_door_meaning_tracks_triangle_direction_not_screen_order_or_color(self):
        for label in ('OPEN','CLOSE'):
            for invert in (False,True):
                image=door_image(label,invert)
                self.assertEqual(experiment.classify_door(image,self.ring)['label'],label)
                self.assertEqual(experiment.classify_door(cv2.flip(image,1),self.ring)['label'],label)

    def test_separator_required_and_digit_or_blank_is_not_door(self):
        image=door_image();image[:,68:73]=70
        self.assertEqual(experiment.classify_door(image,self.ring)['label'],'UNKNOWN')
        for text in ('1','-1','8','11','HOLD'):
            image=np.full((140,140,3),70,np.uint8)
            cv2.putText(image,text,(34,83),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3)
            self.assertEqual(experiment.classify_door(image,self.ring)['label'],'UNKNOWN')
        self.assertEqual(experiment.classify_door(np.zeros((140,140,3),np.uint8),self.ring)['label'],'UNKNOWN')

    def test_one_and_minus_shape_are_only_evidence_and_door_is_not_one(self):
        image=np.full((140,140,3),70,np.uint8)
        cv2.line(image,(73,45),(73,83),(255,255,255),4)
        evidence=experiment.glyph_shape_evidence(image,(70,70),45)
        self.assertEqual(evidence['shape_candidates'],['1'])
        cv2.line(image,(46,63),(60,63),(255,255,255),3)
        evidence=experiment.glyph_shape_evidence(image,(70,70),45)
        self.assertEqual(evidence['shape_candidates'],['-1'])
        self.assertIsNone(evidence['probability']);self.assertFalse(evidence['executable'])
        self.assertEqual(experiment.glyph_shape_evidence(door_image(),(70,70),45)['shape_candidates'],[])

    def test_raw_ocr_objects_remain_unchanged_after_full_analysis(self):
        image=door_image();button=v.ButtonObservation(self.ring)
        result=v.FrameResult(1.,image.shape[:2],(0,0,140,140),'MANUAL_ROI',[button])
        before=result.to_dict()
        record=experiment.analyze(image,result)
        self.assertEqual(result.to_dict(),before)
        self.assertEqual(record['door_symbols'][0]['label'],'OPEN')
        self.assertFalse(record['executable'])

    def test_invalid_or_clipped_patches_are_unknown(self):
        image=np.zeros((140,140,3),np.uint8)
        self.assertEqual(experiment.classify_door(image,v.Ring(70,70,3,1.))['label'],'UNKNOWN')
        self.assertEqual(experiment.glyph_shape_evidence(image,(0,0),30)['status'],'UNKNOWN')
        for x,y,r in ((float('nan'),50,30),(50,float('inf'),30),(50,50,float('nan'))):
            self.assertEqual(experiment.classify_door(image,v.Ring(x,y,r,1.))['label'],'UNKNOWN')
            self.assertEqual(experiment.glyph_shape_evidence(image,(x,y),r)['status'],'UNKNOWN')

    def test_door_misread_as_any_floor_is_not_a_numeric_anchor(self):
        image=door_image()
        for label in ('-1','2','8','11'):
            result=v.FrameResult(1.,image.shape[:2],(0,0,140,140),'MANUAL_ROI',
                                  [v.ButtonObservation(self.ring,label,.999)])
            record=experiment.analyze(image,result)
            self.assertEqual(record['layout']['buttons'][0]['status'],'FUNCTION_OR_WORD')
            self.assertEqual(record['door_symbols'][0]['label'],'OPEN')

    def test_large_photo_feature_resize_maps_geometry_back_to_original_pixels(self):
        image=cv2.resize(door_image(),(2240,2240),interpolation=cv2.INTER_NEAREST)
        ring=v.Ring(1120,1120,720,1.)
        result=v.FrameResult(1.,image.shape[:2],(0,0,2240,2240),'MANUAL_ROI',[v.ButtonObservation(ring)])
        record=experiment.analyze(image,result)
        self.assertEqual(record['door_symbols'][0]['label'],'OPEN')
        self.assertEqual(record['door_symbols'][0]['center_px'],[1120,1120])
        box=record['door_symbols'][0]['glyph_bbox_xyxy']
        self.assertLess(box[0],1120);self.assertGreater(box[2],1120)
        self.assertLess(record['feature_image_scale_xy'][0],1.)

    def test_preview_draws_source_frame_once_and_closes_camera_without_saving(self):
        args=NS(config=ROOT/'config/default.yaml',floor_list=None,inventory_complete=False,ocr_min_score=.7,target_floor=None)
        image=np.full((140,140,3),40,np.uint8);new_image=np.full_like(image,90)
        camera=Mock();camera.get_frame.side_effect=[(image,None),(new_image,None),(new_image,None)]
        detector=Mock();detector.process.return_value=v.FrameResult(99.,image.shape[:2])
        pool=Mock()
        def submit(fn):
            result=fn()
            return NS(done=lambda:True,result=lambda:result)
        pool.submit.side_effect=submit
        with patch('drivers.camera.make_camera',return_value=camera), \
                patch.object(experiment,'ThreadPoolExecutor',return_value=pool), \
                patch.object(experiment.time,'perf_counter',return_value=100.), \
                patch.object(experiment,'draw',side_effect=lambda im,rec:im.copy()) as draw, \
                patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',side_effect=[-1,ord('q')]), \
                patch.object(cv2,'destroyAllWindows'),patch.object(cv2,'imwrite',side_effect=AssertionError('no saving')):
            self.assertEqual(experiment.run_camera(detector,args),0)
        np.testing.assert_array_equal(draw.call_args.args[0],image)
        draw.assert_called_once();camera.close.assert_called_once();pool.shutdown.assert_called_once()

    def test_entry_loads_robot_only_in_explicit_head_factory(self):
        tree=ast.parse(ENTRY.read_text())
        modules=[n.module or '' for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(n.startswith(('elevator.motion','elevator.app','drivers.robot','reBotArm','motorbridge')) for n in modules))
        calls=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertFalse(set(calls)&{'connect','enable','move_to_traj','set_zero_position','send_pos_vel'})


class RecallTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.rings=[v.Ring(x,y,30,1.) for y in (150,250,350) for x in (200,300)]
        self.image=np.full((550,550,3),50,np.uint8)
        for r in self.rings:cv2.circle(self.image,r.center,30,(230,230,230),3)

    def test_guided_search_recovers_real_outer_row_and_checks_current_edges(self):
        added,info=experiment.recover_panel_rims(self.image,self.rings[:4])
        self.assertEqual(len(added),2)
        self.assertEqual(info['guided_rims_added'],2)
        self.assertLessEqual(info['guided_probes'],12)
        for wanted in self.rings[-2:]:self.assertTrue(any(np.linalg.norm(np.subtract(r.center,wanted.center))<4 for r in added))
        occluded=self.image.copy();occluded[310:395,160:340]=50
        added,_=experiment.recover_panel_rims(occluded,self.rings[:4])
        self.assertEqual(added,[])

    def test_grid_predictions_on_blank_or_straight_edges_never_create_a_rim(self):
        image=self.image.copy();image[310:395,160:340]=50
        added,info=experiment.recover_panel_rims(image,self.rings[:4])
        self.assertEqual(added,[]);self.assertEqual(info['guided_rims_added'],0)
        for im in (np.zeros((150,150),np.uint8),np.tile(np.arange(150,dtype=np.uint8),(150,1))):
            distance=v._edge_distance(im)
            self.assertIsNone(experiment.confirm_rim((75,75),25,distance))

    def test_missing_interior_and_upper_buttons_are_recovered_without_floor_layout(self):
        for missing in (0,1,2,3):
            seeds=[r for i,r in enumerate(self.rings) if i!=missing]
            added,_=experiment.recover_panel_rims(self.image,seeds)
            wanted=self.rings[missing]
            self.assertTrue(any(np.linalg.norm(np.subtract(r.center,wanted.center))<4 for r in added))

    def test_search_roi_includes_one_missing_lower_row(self):
        box=experiment.panel_search_box(self.rings[:4],self.image.shape)
        for r in self.rings[-2:]:
            self.assertLess(box[1],r.y-r.radius);self.assertGreater(box[3],r.y+r.radius)

    def test_retry_crops_only_observed_strokes_and_does_not_draw_a_minus(self):
        ring=v.Ring(70,70,45,1.);image=np.full((140,140,3),70,np.uint8)
        cv2.line(image,(73,45),(73,83),(255,255,255),4)
        one,one_box=experiment.isolated_glyph_views(image,ring)
        self.assertEqual(len(one),4)
        cv2.line(image,(46,63),(60,63),(255,255,255),3)
        minus,minus_box=experiment.isolated_glyph_views(image,ring)
        self.assertLess(minus_box[0],one_box[0])
        self.assertFalse(np.array_equal(one[0],minus[0]))
        self.assertEqual(experiment.isolated_glyph_views(np.full_like(image,70),ring),([],None))

    def test_isolated_retry_requires_model_agreement_and_preserves_conflicts(self):
        backend=v._OCRBatch();backend.min_score=.7;backend.engine=NS();backend.input_type=NS
        backend.backend_info={}
        ocr=experiment.RegionOCR(backend)
        r=v.Ring(70,70,45,1.)
        rings=[r]+[v.Ring(70,160+100*i,45,1.) for i in range(3)]
        entries=[dict(glyph={'shape_candidates':['1']},best=None,ring=ring) for ring in rings]
        original=[v.ButtonObservation(r,variants=[('Y',.9),('1',.75)])]+[
            v.ButtonObservation(ring,str(i+2),.99) for i,ring in enumerate(rings[1:])]
        frame=Mock(stats={'inferred_patches':4,'batches':1})
        frame.recognize.return_value=[('1',.9),('1',.95),('1',.96),('1',.97)]
        views=[np.zeros((30,30,3),np.uint8)]*4
        with patch.object(v,'_FrameRecognition',return_value=frame),patch.object(recognition,'isolated_glyph_views',return_value=(views,(60,40,80,90))):
            inferred=ocr.retry_unread(np.zeros((500,500,3),np.uint8),rings,list(original),entries,list(range(4)))
            self.assertEqual(inferred[0].text,'1')
            self.assertEqual(inferred[0].text_region_source,'isolated_glyph_retry')
            self.assertEqual(ocr.retry_audit['0']['original_variants'],[('Y',.9),('1',.75)])
            conflict=list(original);conflict[0]=v.ButtonObservation(r,variants=[('1',.9),('-1',.95)])
            rejected=ocr.retry_unread(np.zeros((500,500,3),np.uint8),rings,conflict,entries,list(range(4)))
            self.assertIsNone(rejected[0].text)
            self.assertEqual(frame.recognize.call_count,1)

    def test_single_fresh_retry_needs_two_prior_reads_same_pixels_and_no_conflict(self):
        backend=v._OCRBatch();backend.min_score=.7;backend.engine=NS();backend.input_type=NS;backend.backend_info={}
        ocr=experiment.RegionOCR(backend);ring=v.Ring(70,70,45,1.)
        rings=[ring]+[v.Ring(70,160+100*i,45,1.) for i in range(3)]
        entries=[dict(glyph={'shape_candidates':['1']},best=None,ring=r) for r in rings]
        entries[0].update(best=v.ButtonObservation(ring,'1',.95),count=2,same_pixels=True,last_good=0.)
        original=[v.ButtonObservation(ring,variants=[('Y',.9)])]+[
            v.ButtonObservation(r,str(i+2),.99) for i,r in enumerate(rings[1:])]
        frame=Mock(stats={'inferred_patches':2,'batches':1})
        views=[np.zeros((30,30,3),np.uint8)]*2
        with patch.object(v,'_FrameRecognition',return_value=frame),patch.object(recognition,'isolated_glyph_views',return_value=(views,(60,40,80,90))):
            frame.recognize.return_value=[('1',.94),('I',.9)]
            accepted=ocr.retry_unread(np.zeros((500,500,3),np.uint8),rings,list(original),entries,list(range(4)))
            self.assertEqual(accepted[0].text,'1')
            self.assertTrue(ocr.retry_audit['0']['used_prior_verified_ocr'])
            self.assertAlmostEqual(accepted[0].confidence,.94)
            for count,same,readings in ((1,True,[('1',.94),('I',.9)]),
                                        (2,False,[('1',.94),('I',.9)]),
                                        (2,True,[('1',.79),('I',.9)]),
                                        (2,True,[('1',.94),('-1',.92)])):
                entries[0].update(count=count,same_pixels=same);frame.recognize.return_value=readings
                rejected=ocr.retry_unread(np.zeros((500,500,3),np.uint8),rings,list(original),entries,list(range(4)))
                self.assertIsNone(rejected[0].text)

    def test_retry_budget_is_two_regions_and_stays_in_observed_numeric_columns(self):
        backend=v._OCRBatch();backend.min_score=.7;backend.engine=NS();backend.input_type=NS;backend.backend_info={}
        ocr=experiment.RegionOCR(backend)
        rings=[v.Ring(70,100+100*i,30,1.) for i in range(6)]+[v.Ring(300,200,30,1.)]
        entries=[dict(glyph={'shape_candidates':['1']},best=None,ring=r) for r in rings]
        observations=[v.ButtonObservation(r,str(i+2),.99) if i<3 else v.ButtonObservation(r) for i,r in enumerate(rings)]
        frame=Mock(stats={'inferred_patches':8,'batches':1});frame.recognize.return_value=[('1',.95)]*8
        with patch.object(v,'_FrameRecognition',return_value=frame),patch.object(recognition,'isolated_glyph_views',return_value=([np.zeros((20,20,3),np.uint8)]*4,(60,40,80,90))):
            result=ocr.retry_unread(np.zeros((800,500,3),np.uint8),rings,observations,entries,list(range(len(rings))))
        self.assertEqual(ocr.retry_stats['glyph_retry_buttons'],2)
        self.assertEqual(len(frame.recognize.call_args.args[0]),8)
        self.assertIsNone(result[-1].text)


class HeadPreviewTests(unittest.TestCase):
    def setup_preview(self):
        args=experiment.parse_args(['--camera','--head-up'])
        image=np.zeros((140,140,3),np.uint8)
        camera=Mock();camera.get_frame.return_value=(image,None)
        detector=Mock();detector.process.return_value=v.FrameResult(99.,image.shape[:2])
        robot=Mock()
        options=dict(startup_view=dict(joints_deg=[0,0,-10,-20,0,0],duration_s=2.,wait_for_arrival=False),
                     gripper=dict(initialize=True,closed_angle_deg=0.,closed_tolerance_rad=.01))
        pool=Mock();pool.submit.side_effect=lambda fn:NS(done=lambda:True,result=fn)
        return args,camera,detector,robot,options,pool

    def test_head_flag_requires_camera_and_default_remains_no_robot(self):
        self.assertFalse(experiment.parse_args(['--camera']).head_up)
        for argv in (['--head-up'],['--head-up','--images','x.jpg']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):experiment.parse_args(argv)

    def test_head_factory_reads_current_workflow_without_starting_robot(self):
        from elevator.app import load_workflow
        args=experiment.parse_args(['--camera','--head-up']);cfg=v.load_config(args.config)
        original=repr(cfg)
        with patch('elevator.motion.Robot') as constructor:
            robot,options=experiment.make_head_robot(cfg,args)
        self.assertIs(robot,constructor.return_value);robot.start.assert_not_called()
        self.assertEqual(repr(cfg),original)
        self.assertEqual(options,load_workflow(args.workflow_config))
        self.assertEqual(constructor.call_args.args[0]['trajectory'],options['trajectory'])

    def test_reuses_configured_head_once_then_closes_without_button_motion(self):
        args,camera,detector,robot,options,pool=self.setup_preview()
        robot.head_up.side_effect=lambda goals,duration,cancelled,**kwargs:self.assertFalse(cancelled())
        with patch('drivers.camera.make_camera',return_value=camera), \
                patch.object(experiment,'make_head_robot',return_value=(robot,options)), \
                patch.object(experiment,'ThreadPoolExecutor',return_value=pool), \
                patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',side_effect=[-1,ord('q')]),patch.object(cv2,'destroyAllWindows'):
            self.assertEqual(experiment.run_camera(detector,args),0)
        robot.start.assert_called_once();robot.head_up.assert_called_once()
        self.assertEqual(robot.head_up.call_args.args[:2],([0,0,-10,-20,0,0,0.],2.))
        self.assertFalse(robot.head_up.call_args.kwargs['wait_for_arrival'])
        robot.close.assert_called_once_with(home=False);robot.move.assert_not_called()
        detector.request_refresh.assert_called_once();camera.close.assert_called_once();pool.shutdown.assert_called_once()

    def test_start_failure_still_disables_and_releases_camera(self):
        args,camera,detector,robot,options,pool=self.setup_preview();robot.start.side_effect=RuntimeError('bus unavailable')
        with patch('drivers.camera.make_camera',return_value=camera), \
                patch.object(experiment,'make_head_robot',return_value=(robot,options)), \
                patch.object(experiment,'ThreadPoolExecutor',return_value=pool),patch.object(cv2,'destroyAllWindows'), \
                self.assertRaisesRegex(RuntimeError,'bus unavailable'):
            experiment.run_camera(detector,args)
        robot.close.assert_called_once_with(home=False);camera.close.assert_called_once();pool.shutdown.assert_called_once()
        robot.head_up.assert_not_called()

    def test_q_during_head_cancels_without_starting_ocr(self):
        args,camera,detector,robot,options,pool=self.setup_preview()
        def head(goals,duration,cancelled,**kwargs):
            self.assertTrue(cancelled());raise RuntimeError('cancelled')
        robot.head_up.side_effect=head
        with patch('drivers.camera.make_camera',return_value=camera), \
                patch.object(experiment,'make_head_robot',return_value=(robot,options)), \
                patch.object(experiment,'ThreadPoolExecutor',return_value=pool),patch.object(cv2,'imshow'), \
                patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'):
            self.assertEqual(experiment.run_camera(detector,args),0)
        robot.close.assert_called_once_with(home=False);pool.submit.assert_not_called()

    def test_cleanup_error_is_reported_and_all_other_cleanup_still_runs(self):
        args,camera,detector,robot,options,pool=self.setup_preview();robot.close.side_effect=RuntimeError('disable not confirmed')
        with patch('drivers.camera.make_camera',return_value=camera), \
                patch.object(experiment,'make_head_robot',return_value=(robot,options)), \
                patch.object(experiment,'ThreadPoolExecutor',return_value=pool),patch.object(cv2,'imshow'), \
                patch.object(cv2,'waitKey',return_value=ord('q')),patch.object(cv2,'destroyAllWindows'), \
                self.assertRaisesRegex(RuntimeError,'disable not confirmed'):
            experiment.run_camera(detector,args)
        camera.close.assert_called_once();pool.shutdown.assert_called_once()


if __name__=='__main__':unittest.main()
