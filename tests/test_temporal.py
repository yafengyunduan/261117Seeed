"""Offline checks for the current implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from elevator import temporal

def frame(at, labels=("3", "4", None, None, None, None), *, dx=0, source="GEOMETRY_PROPOSAL",
          confidence=.95, omit=(), radius=18., panel=True):
    buttons = []
    for index, label in enumerate(labels):
        if index in omit:
            continue
        x, y = 100 + (index % 2)*100 + dx, 100 + (index // 2)*80
        buttons.append(SimpleNamespace(ring=SimpleNamespace(x=x, y=y, radius=radius, edge_support=.97),
                                       text=label, confidence=confidence if label is not None else None,
                                       text_bbox=(x-5, y-5, x+5, y+5) if label else None,
                                       status="UNSTABLE" if label else "NO_TEXT", observations=0))
    return SimpleNamespace(captured_at=float(at), image_shape=(480, 640),
                           panel_bbox=(60+dx, 60, 250+dx, 320) if panel else None,
                           panel_source=source if panel else "NO_PANEL", buttons=buttons, errors=[])


def prime(tracker, **kwargs):
    for at in (1, 2, 3):
        result = frame(at, **kwargs)
        summary = tracker.update(result, now=at)
    return result, summary


class TemporalChecks(unittest.TestCase):
    def test_01_three_distinct_current_ocr_frames_required(self):
        tracker = temporal.TemporalButtonTracker()
        for at, expected in [(1, "UNSTABLE"), (2, "UNSTABLE"), (3, "STABLE")]:
            result = frame(at)
            tracker.update(result, now=at)
            self.assertEqual(result.buttons[0].status, expected)
        self.assertEqual(result.buttons[0].observations, 3)

    def test_02_repeated_timestamp_does_not_count(self):
        tracker = temporal.TemporalButtonTracker()
        for _ in range(5):
            result = frame(1)
            tracker.update(result, now=1)
            self.assertNotEqual(result.buttons[0].status, "STABLE")
        self.assertEqual(len(next(iter(tracker.tracks.values())).votes), 1)

    def test_03_old_timestamp_does_not_count(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(2)
        summary = tracker.update(result, now=3.1)
        self.assertFalse(summary.accepted_frame)
        self.assertFalse(any(b.status == "STABLE" for b in result.buttons))

    def test_04_blank_ocr_never_inherits_cached_label(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(4, (None,)*6)
        summary = tracker.update(result, now=4)
        self.assertTrue(all(b.text is None and b.status == "NO_TEXT" for b in result.buttons))
        self.assertEqual({c.label for c in summary.cached_labels}, {"3", "4"})
        self.assertTrue(all(not c.valid_current for c in summary.cached_labels))

    def test_05_low_confidence_has_no_current_target(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(4, confidence=.2)
        summary = tracker.update(result, now=4)
        self.assertEqual(result.buttons[0].text, "3")
        self.assertEqual(result.buttons[0].status, "LOW_CONF")
        self.assertTrue(all(not c.valid_current for c in summary.cached_labels))

    def test_06_label_conflict_keeps_entity_id_but_unstable(self):
        tracker = temporal.TemporalButtonTracker()
        original, _ = prime(tracker)
        track_id = original.buttons[0].track_id
        result = frame(4, ("8", "4", None, None, None, None))
        summary = tracker.update(result, now=4)
        self.assertEqual(result.buttons[0].track_id, track_id)
        self.assertEqual(result.buttons[0].text, "8")
        self.assertEqual(result.buttons[0].status, "UNSTABLE")
        cached = next(c for c in summary.cached_labels if c.track_id == track_id)
        self.assertEqual(cached.label, "3")
        self.assertFalse(cached.valid_current)

    def test_07_change_label_requires_new_majority(self):
        tracker = temporal.TemporalButtonTracker(max_age_s=10)
        prime(tracker)
        for at in (4, 5, 6):
            result = frame(at, ("8", "4", None, None, None, None))
            tracker.update(result, now=at)
            self.assertEqual(result.buttons[0].status, "UNSTABLE")
        result = frame(7, ("8", "4", None, None, None, None))
        summary = tracker.update(result, now=7)
        self.assertEqual(result.buttons[0].status, "STABLE")
        self.assertEqual(summary.cached_labels[0].label, "8")

    def test_08_single_bad_read_needs_reconfirmation(self):
        tracker = temporal.TemporalButtonTracker(max_age_s=10)
        prime(tracker)
        tracker.update(frame(4, ("8", "4", None, None, None, None)), now=4)
        for at in (5, 6, 7):
            result = frame(at)
            tracker.update(result, now=at)
            self.assertEqual(result.buttons[0].status, "STABLE" if at == 7 else "UNSTABLE")

    def test_09_panel_missing_once_preserves_track_history(self):
        tracker = temporal.TemporalButtonTracker(max_age_s=8)
        old, _ = prime(tracker)
        lost = frame(4, labels=(), panel=False)
        summary = tracker.update(lost, now=4)
        self.assertEqual(len(summary.cached_labels), 2)
        self.assertTrue(all(not c.valid_current for c in summary.cached_labels))
        result = frame(5)
        tracker.update(result, now=5)
        self.assertEqual(result.buttons[0].track_id, old.buttons[0].track_id)
        self.assertEqual(result.buttons[0].status, "STABLE")

    def test_10_one_missing_entity_does_not_clear_others(self):
        tracker = temporal.TemporalButtonTracker(max_age_s=8)
        old, _ = prime(tracker)
        result = frame(4, omit=(1,))
        summary = tracker.update(result, now=4)
        self.assertEqual(result.buttons[0].status, "STABLE")
        cached = next(c for c in summary.cached_labels if c.label == "4")
        self.assertFalse(cached.valid_current)
        restored = frame(5)
        tracker.update(restored, now=5)
        self.assertEqual(restored.buttons[1].track_id, old.buttons[1].track_id)

    def test_11_short_ocr_gap_preserves_evidence(self):
        tracker = temporal.TemporalButtonTracker(max_age_s=8)
        old, _ = prime(tracker)
        tracker.update(frame(4, (None,)*6), now=4)
        result = frame(5)
        tracker.update(result, now=5)
        self.assertEqual(result.buttons[0].track_id, old.buttons[0].track_id)
        self.assertEqual(result.buttons[0].status, "STABLE")

    def test_12_aged_frame_and_cache_are_invalid(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(3)
        summary = tracker.update(result, now=10)
        self.assertFalse(summary.accepted_frame)
        self.assertEqual(summary.cached_labels, ())
        self.assertFalse(any(b.status == "STABLE" for b in result.buttons))

    def test_13_miss_limit_expires_tracks(self):
        tracker = temporal.TemporalButtonTracker(max_age_s=20, max_missed=2)
        prime(tracker)
        for at in (4, 5, 6):
            tracker.update(frame(at, labels=(), panel=False), now=at)
        self.assertEqual(tracker.summary.cached_labels, ())
        result = frame(7)
        tracker.update(result, now=7)
        self.assertEqual(result.buttons[0].observations, 1)

    def test_14_panel_geometry_needs_no_ocr(self):
        evidence = temporal.PanelEvidence()
        for at in (1, 2, 3):
            state = evidence.update(frame(at, (None,)*6), now=at)
        self.assertTrue(state.stable and state.valid_current)

    def test_15_panel_geometry_rejects_fewer_than_four_rims(self):
        evidence = temporal.PanelEvidence()
        state = evidence.update(frame(1, labels=(None,)*3), now=1)
        self.assertFalse(state.candidate)

    def test_16_panel_missing_soft_state_has_no_current_validity(self):
        evidence = temporal.PanelEvidence(max_age_s=8)
        for at in (1, 2, 3):
            evidence.update(frame(at), now=at)
        state = evidence.update(frame(4, labels=(), panel=False), now=4)
        self.assertTrue(state.stable)
        self.assertFalse(state.valid_current)
        self.assertFalse(evidence.stable)
        state = evidence.update(frame(5), now=5)
        self.assertTrue(state.stable and state.valid_current)

    def test_17_generation_change_resets(self):
        tracker = temporal.TemporalButtonTracker()
        for at in (1, 2, 3):
            tracker.update(frame(at), generation=0, now=at)
        result = frame(4)
        summary = tracker.update(result, generation=1, now=4)
        self.assertEqual(summary.reset_reason, "GENERATION_CHANGED")
        self.assertEqual(result.buttons[0].observations, 1)

    def test_18_camera_pose_movement_resets(self):
        tracker = temporal.TemporalButtonTracker()
        for at in (1, 2, 3):
            tracker.update(frame(at), camera_pose=np.eye(4), now=at)
        moved = np.eye(4)
        moved[0, 3] = .02
        result = frame(4)
        summary = tracker.update(result, camera_pose=moved, now=4)
        self.assertEqual(summary.reset_reason, "CAMERA_MOVED")
        self.assertEqual(result.buttons[0].status, "UNSTABLE")

    def test_19_large_panel_movement_resets(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(4, dx=90)
        summary = tracker.update(result, now=4)
        self.assertEqual(summary.reset_reason, "PANEL_LAYOUT_MOVED")
        self.assertEqual(result.buttons[0].observations, 1)

    def test_20_box_margin_change_with_same_rims_keeps_ids(self):
        tracker = temporal.TemporalButtonTracker()
        old, _ = prime(tracker)
        result = frame(4, source="YOLOE")
        result.panel_bbox = (10, 10, 400, 400)
        tracker.update(result, now=4)
        self.assertEqual(result.buttons[0].track_id, old.buttons[0].track_id)
        self.assertEqual(result.buttons[0].status, "STABLE")

    def test_21_ordering_does_not_change_identity(self):
        tracker = temporal.TemporalButtonTracker()
        old, _ = prime(tracker)
        result = frame(4)
        result.buttons.reverse()
        tracker.update(result, now=4)
        self.assertEqual(result.buttons[-1].track_id, old.buttons[0].track_id)

    def test_22_cached_age_is_explicit(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        cached = tracker.cached_labels(now=4)
        self.assertTrue(all(c.age_s == 1 and not c.valid_current for c in cached))
        self.assertEqual(tracker.cached_labels(now=9), ())

    def test_23_duplicate_result_cannot_keep_prior_stable_status(self):
        tracker = temporal.TemporalButtonTracker()
        result, _ = prime(tracker)
        self.assertEqual(result.buttons[0].status, "STABLE")
        summary = tracker.update(result, now=3.1)
        self.assertFalse(summary.accepted_frame)
        self.assertNotEqual(result.buttons[0].status, "STABLE")

    def test_24_weak_rim_support_has_no_current_target(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(4)
        result.buttons[0].ring.edge_support = .5
        tracker.update(result, now=4)
        self.assertEqual(result.buttons[0].status, "UNSTABLE")

    def test_25_no_old_label_overwrites_conflicting_current_ocr(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(4, ("8", None, None, None, None, None))
        tracker.update(result, now=4)
        self.assertEqual([b.text for b in result.buttons], ["8", None, None, None, None, None])

    def test_26_errors_do_not_create_current_target(self):
        tracker = temporal.TemporalButtonTracker()
        prime(tracker)
        result = frame(4)
        result.buttons[0].status = "OCR_ERROR"
        tracker.update(result, now=4)
        self.assertEqual(result.buttons[0].status, "OCR_ERROR")

    def test_27_invalid_parameters_rejected(self):
        for kwargs in ({"required": 2}, {"max_age_s": float("nan")}, {"vote_window": 2}, {"min_vote_fraction": .5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                temporal.TemporalButtonTracker(**kwargs)

    def test_28_default_vision_compatibility_alias(self):
        tracker = temporal.StabilityTracker(required=3, max_age=4)
        self.assertIsInstance(tracker, temporal.TemporalButtonTracker)
        tracker.update(frame(1), now=1)


class LabelMemoryChecks(unittest.TestCase):
    def tracker(self, **kwargs):
        tracker = temporal.TemporalButtonTracker(memory_age_s=10., **kwargs)
        prime(tracker)
        return tracker

    def blank(self, tracker, at=4, **kwargs):
        result = frame(at, labels=(None,)*6, **kwargs)
        tracker.update(result, now=at)
        return result

    def test_confirmed_labels_bridge_a_gap_without_fabricating_current_ocr(self):
        tracker = self.tracker()
        for at in range(4, 9):
            result = self.blank(tracker, at)
        evidence = tracker.remembered_associations(result, now=8)
        self.assertEqual({e.label for e in evidence}, {'3','4'})
        self.assertEqual(evidence[0].frames_ago, 5)
        self.assertEqual(evidence[0].confirmed_at, 3.)
        self.assertTrue(all(b.text is None and b.status == 'NO_TEXT' for b in result.buttons))
        self.assertTrue(all(not c.valid_current for c in tracker.cached_labels(now=8)))

    def test_wall_clock_and_processed_frame_budgets_expire_independently(self):
        tracker = self.tracker()
        for at in range(4, 14):
            result = self.blank(tracker, at)
        self.assertTrue(tracker.remembered_associations(result, now=13))
        self.assertEqual(tracker.remembered_associations(result, now=13.01), ())
        self.assertEqual(tracker.cached_labels(now=13.01), ())
        tracker = self.tracker(memory_frames=3)
        for at in (4,5,6):
            result = self.blank(tracker, at)
        self.assertTrue(tracker.remembered_associations(result, now=6))
        result = self.blank(tracker, 7)
        self.assertEqual(tracker.remembered_associations(result, now=7), ())

    def test_single_or_two_readings_do_not_create_confirmed_memory(self):
        tracker = temporal.TemporalButtonTracker(memory_age_s=10.)
        for at in (1,2):
            tracker.update(frame(at), now=at)
        result = self.blank(tracker, 3)
        self.assertEqual(tracker.cached_labels(now=3), ())
        self.assertEqual(tracker.remembered_associations(result, now=3), ())

    def test_conflicting_numeric_read_blocks_memory_until_new_consensus(self):
        for confidence in (.95, .4):
            tracker = self.tracker()
            conflict = frame(4, labels=('8','4',None,None,None,None), confidence=confidence)
            tracker.update(conflict, now=4)
            result = self.blank(tracker, 5)
            self.assertNotIn('3', [e.label for e in tracker.remembered_associations(result, now=5)])
            self.assertIn('3', [e.label for e in tracker.cached_labels(now=5)])
            for at in (6,7,8):
                tracker.update(frame(at), now=at)
            result = self.blank(tracker, 9)
            self.assertIn('3', [e.label for e in tracker.remembered_associations(result, now=9)])

    def test_errors_missing_weak_ambiguous_or_drifted_rims_cannot_use_memory(self):
        for issue in ('global_error','ocr_error','nonfinite','missing','weak','duplicate','drift'):
            tracker = self.tracker()
            result = frame(4, labels=(None,)*6, dx=6 if issue == 'drift' else 0,
                           omit=(0,) if issue == 'missing' else ())
            if issue == 'global_error': result.errors.append('OCR_ERROR')
            elif issue == 'ocr_error': result.buttons[0].status = 'OCR_ERROR'
            elif issue == 'nonfinite': result.buttons[0].confidence = float('nan')
            elif issue == 'weak': result.buttons[0].ring.edge_support = .5
            elif issue == 'duplicate':
                import copy
                result.buttons.append(copy.deepcopy(result.buttons[0]))
            tracker.update(result, now=4)
            with self.subTest(issue=issue):
                self.assertNotIn('3', [e.label for e in tracker.remembered_associations(result, now=4)])

    def test_motion_generation_shape_or_camera_pose_prevents_reusing_identity(self):
        for issue in ('generation','shape','pose'):
            tracker = temporal.TemporalButtonTracker(memory_age_s=10.)
            for at in (1,2,3):
                tracker.update(frame(at), generation=0, camera_pose=np.eye(4), now=at)
            result = frame(4, labels=(None,)*6)
            pose = np.eye(4)
            if issue == 'shape': result.image_shape = (481,640)
            if issue == 'pose': pose[0,3] = .004  # Below panel reset's 1cm, above identity's 3mm.
            tracker.update(result, generation=1 if issue == 'generation' else 0, camera_pose=pose, now=4)
            with self.subTest(issue=issue):
                self.assertEqual(tracker.remembered_associations(result, now=4), ())

    def test_reordering_matches_current_entities_and_duplicate_frames_add_no_evidence(self):
        tracker = self.tracker()
        result = frame(4, labels=(None,)*6)
        result.buttons.reverse()
        tracker.update(result, now=4)
        association = next(e for e in tracker.remembered_associations(result, now=4) if e.label == '3')
        self.assertEqual(association.observation_index, 5)
        self.assertEqual(association.ring_xyr, (100.,100.,18.))
        tracker.update(result, now=4.1)
        self.assertEqual(tracker.remembered_associations(result, now=4.1), ())
        self.assertEqual(tracker.cached_labels(now=4.1)[0].frames_ago, 1)

    def test_eighty_percent_boundary_is_used_by_temporal_confirmation(self):
        for score, stable in ((.80,True),(.799,False)):
            tracker = temporal.TemporalButtonTracker()
            result, _ = prime(tracker, confidence=score)
            self.assertEqual(result.buttons[0].status == 'STABLE', stable)

    def test_invalid_cache_limits_are_rejected(self):
        for kwargs in ({'memory_age_s':0}, {'memory_age_s':31}, {'memory_age_s':float('nan')},
                       {'memory_frames':0}, {'memory_frames':121}, {'memory_frames':1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                temporal.TemporalButtonTracker(**kwargs)



class SparseWorkflowEvidenceChecks(unittest.TestCase):
    def make(self,at,source='CURRENT_OCR',*,label='3',identity=1,pixels=True):
        result=frame(at)
        result.tracking=dict(workflow_evidence=True,region_evidence={},display_label_cache={})
        for i,b in enumerate(result.buttons):
            kind=source if i==0 else ('CURRENT_OCR' if b.text is not None else 'NO_TEXT')
            result.tracking['region_evidence'][str(i)]=dict(source=kind,pixel_match=pixels,
                track_id=identity if i==0 else i+10,ocr_observations=3)
        b=result.buttons[0];b.text=label;b.confidence=.95;b.status='UNSTABLE'
        if source!='CURRENT_OCR':
            b.text=b.confidence=b.text_bbox=None;b.status='NO_TEXT'
        if source=='PIXEL_MATCHED_CACHE':
            result.tracking['display_label_cache']['0']=dict(text=label,confidence=.95,status='STABLE_CACHE')
        return result

    def update(self,tracker,result,generation=0):
        tracker.update(result,generation,camera_pose=np.eye(4),now=result.captured_at)
        return result

    def prime(self,tracker):
        for at in (1.,1.2,1.4):result=self.update(tracker,self.make(at))
        return result

    def test_display_cache_cannot_turn_one_model_read_into_three_votes(self):
        tracker=temporal.TemporalButtonTracker(memory_age_s=10.)
        self.update(tracker,self.make(1.))
        for k in range(1,9):
            result=self.update(tracker,self.make(1.+k*.1,'PIXEL_MATCHED_CACHE'))
        track=tracker.tracks[result.buttons[0].track_id]
        self.assertEqual(len(track.votes),1);self.assertIsNone(track.confirmed_label)
        self.assertNotIn('3',[c.label for c in tracker.cached_labels(now=1.8)])
        self.update(tracker,self.make(2.));result=self.update(tracker,self.make(2.2))
        self.assertEqual(result.buttons[0].status,'STABLE')
        cached=self.update(tracker,self.make(2.4,'PIXEL_MATCHED_CACHE'))
        self.assertIsNone(cached.buttons[0].text)
        self.assertEqual(cached.buttons[0].status,'NO_TEXT')
        self.assertIn('3',[a.label for a in tracker.remembered_associations(cached,now=2.4)])

    def test_skipped_ocr_frames_allow_three_real_reads_across_refresh_intervals(self):
        tracker=temporal.TemporalButtonTracker(memory_age_s=10.)
        for at in (.1,2.2,4.3):result=self.update(tracker,self.make(at))
        self.assertEqual(result.buttons[0].observations,3)
        self.assertEqual(result.buttons[0].status,'STABLE')
        self.assertEqual(len(tracker.tracks[result.buttons[0].track_id].votes),3)

    def test_changed_pixels_or_missing_proof_cannot_bind_old_label(self):
        for source,pixels,identity in (('PIXEL_MATCHED_CACHE',False,1),('NO_TEXT',True,1),('CURRENT_OCR',True,2)):
            tracker=temporal.TemporalButtonTracker(memory_age_s=10.);self.prime(tracker)
            result=self.update(tracker,self.make(1.6,source,pixels=pixels,identity=identity))
            self.assertNotIn('3',[a.label for a in tracker.remembered_associations(result,now=1.6)])
            if identity==2:
                self.assertEqual(result.buttons[0].observations,1)
                self.assertEqual(result.buttons[0].status,'UNSTABLE')

    def test_changed_cached_digit_blocks_old_identity_without_creating_new_votes(self):
        tracker=temporal.TemporalButtonTracker(memory_age_s=10.);self.prime(tracker)
        result=self.update(tracker,self.make(1.6,'PIXEL_MATCHED_CACHE',label='8'))
        self.assertNotIn('3',[a.label for a in tracker.remembered_associations(result,now=1.6)])
        track=tracker.tracks[result.buttons[0].track_id]
        self.assertTrue(track.label_conflict);self.assertEqual({v[1] for v in track.votes},{'3'})
        result=self.update(tracker,self.make(1.8,'PIXEL_MATCHED_CACHE'))
        self.assertNotIn('3',[a.label for a in tracker.remembered_associations(result,now=1.8)])
        for at in (2.,2.2,2.4):result=self.update(tracker,self.make(at))
        self.assertEqual(result.buttons[0].status,'STABLE')

    def test_motion_generation_resets_both_votes_and_confirmed_label(self):
        tracker=temporal.TemporalButtonTracker(memory_age_s=10.);self.prime(tracker)
        result=self.update(tracker,self.make(1.6,'PIXEL_MATCHED_CACHE'),generation=1)
        self.assertEqual(tracker.cached_labels(now=1.6),())
        self.assertEqual(tracker.remembered_associations(result,now=1.6),())

    def test_consumer_does_not_count_text_fields_tagged_as_display_cache(self):
        tracker=temporal.TemporalButtonTracker(memory_age_s=10.)
        self.update(tracker,self.make(1.))
        for at in (1.2,1.4):
            result=self.make(at,'PIXEL_MATCHED_CACHE');result.buttons[0].text='3';result.buttons[0].confidence=.95
            self.update(tracker,result)
        self.assertNotEqual(result.buttons[0].status,'STABLE')
        self.assertEqual(len(tracker.tracks[result.buttons[0].track_id].votes),1)
