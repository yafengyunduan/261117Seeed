"""Temporal evidence for live panel/rim observations; no camera or SDK access.

Identity follows measured rim geometry, never the OCR label. Cached labels are
returned separately and are never copied into result.buttons. A current button
can be STABLE only when its own current OCR supports the voted label.
An explicit LabelAssociation may bridge an OCR gap using the same currently
measured rim; it carries no cached image, depth or three-dimensional target.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Optional
import unicodedata

import numpy as np


def _clock(now: Optional[float]) -> float:
    value = time.perf_counter() if now is None else float(now)
    if not math.isfinite(value):
        raise ValueError("now 必须有限")
    return value


def _shape(result: Any) -> Optional[tuple[int, int]]:
    shape = tuple(getattr(result, "image_shape", ()))
    return shape if len(shape) == 2 and all(isinstance(v, (int, np.integer)) and v > 0 for v in shape) else None


def _box(result: Any) -> Optional[np.ndarray]:
    shape = _shape(result)
    raw = getattr(result, "panel_bbox", None)
    if shape is None or raw is None:
        return None
    box = np.asarray(raw, dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)) or np.any(box[2:] <= box[:2]):
        return None
    if min(box[:2]) < 0 or box[2] > shape[1] or box[3] > shape[0]:
        return None
    return box


def _ring(observation: Any, shape: Optional[tuple[int, int]]) -> Optional[np.ndarray]:
    ring = getattr(observation, "ring", None)
    if ring is None or shape is None:
        return None
    values = np.asarray([getattr(ring, "x", math.nan), getattr(ring, "y", math.nan),
                         getattr(ring, "radius", math.nan), getattr(ring, "edge_support", math.nan)], float)
    x, y, radius, support = values
    if not np.all(np.isfinite(values)) or not 0 <= x < shape[1] or not 0 <= y < shape[0] or radius < 4 or not 0 <= support <= 1:
        return None
    return values


def _geometry(result: Any, min_support: float = .75) -> list[np.ndarray]:
    box, shape = _box(result), _shape(result)
    rings = [_ring(button, shape) for button in getattr(result, "buttons", ())]
    return [r for r in rings if r is not None and r[3] >= min_support and box is not None
            and box[0] <= r[0] <= box[2] and box[1] <= r[1] <= box[3]]


def panel_candidate(result: Any, min_geometry_rings: int = 4) -> bool:
    """YOLO panel or repeated measured rims, independently of OCR success."""
    box = _box(result)
    if box is None:
        return False
    source = getattr(result, "panel_source", "NO_PANEL")
    if source in {"YOLOE", "MANUAL_ROI"}:
        return True
    if source not in {"GEOMETRY_PROPOSAL", "GEOMETRY"}:
        return False
    rings = _geometry(result)
    if len(rings) < min_geometry_rings:
        return False
    radius = float(np.median([r[2] for r in rings]))
    # Reject duplicate/nested circles and a cluster occupying one physical rim.
    distinct = []
    for ring in rings:
        if .55 <= ring[2] / radius <= 1.7 and all(np.linalg.norm(ring[:2]-old[:2]) >= 1.3*min(ring[2], old[2]) for old in distinct):
            distinct.append(ring)
    if len(distinct) < min_geometry_rings:
        return False
    centers = np.array([r[:2] for r in distinct])
    return bool(np.linalg.norm(np.ptp(centers, axis=0)) >= 4*radius)


def _layout_matches(first: list[np.ndarray], second: list[np.ndarray]) -> bool:
    if min(len(first), len(second)) < 4:
        return False
    used, matched = set(), 0
    for ring in second:
        candidates = [(np.linalg.norm(ring[:2]-old[:2]), index) for index, old in enumerate(first)
                      if index not in used and .65 <= ring[2]/old[2] <= 1.5]
        if not candidates:
            continue
        distance, index = min(candidates)
        if distance <= max(6., .8*max(ring[2], first[index][2])):
            used.add(index)
            matched += 1
    return matched >= max(3, math.ceil(.6*min(len(first), len(second))))


def _box_consistent(first: np.ndarray, second: np.ndarray) -> bool:
    lo, hi = np.maximum(first[:2], second[:2]), np.minimum(first[2:], second[2:])
    overlap = float(np.prod(np.maximum(0., hi-lo)))
    union = float(np.prod(first[2:]-first[:2]) + np.prod(second[2:]-second[:2]) - overlap)
    delta = np.linalg.norm((first[:2]+first[2:]-second[:2]-second[2:])*.5)
    return union > 0 and overlap/union >= .35 and delta <= max(6., .15*np.linalg.norm(first[2:]-first[:2]))


def _camera_changed(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> bool:
    if first is None or second is None:
        return False
    relative = first[:3, :3].T @ second[:3, :3]
    angle = math.acos(float(np.clip((np.trace(relative)-1)*.5, -1, 1)))
    return np.linalg.norm(first[:3, 3]-second[:3, 3]) > .01 or angle > math.radians(3)


def _pose(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    T = np.asarray(value, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("camera_pose 必须为有限4x4变换")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5):
        raise ValueError("camera_pose 旋转部分无效")
    return T.copy()


@dataclass(frozen=True)
class PanelState:
    stable: bool
    valid_current: bool
    accepted_frame: bool
    count: int
    missed: int
    age_s: float
    candidate: bool
    reset_reason: Optional[str] = None


class PanelEvidence:
    """Short missing observations retain history, never current validity."""
    def __init__(self, required: int = 3, max_age_s: float = 4., max_missed: int = 3):
        if int(required) != required or required < 3 or not math.isfinite(max_age_s) or max_age_s <= 0 or int(max_missed) != max_missed or max_missed < 0:
            raise ValueError("PanelEvidence 要求 required>=3、有限max_age_s>0、max_missed>=0")
        self.required, self.max_age_s, self.max_missed = int(required), float(max_age_s), int(max_missed)
        self.reset()

    def reset(self) -> None:
        self.count = self.missed = 0
        self._last_frame_at = self._last_seen_at = None
        self._anchor_box = self._anchor_pose = None
        self._anchor_rings: list[np.ndarray] = []
        self._generation = self._image_shape = None
        self.summary = PanelState(False, False, False, 0, 0, math.inf, False)

    @property
    def stable(self) -> bool:
        return self.summary.stable and self.summary.valid_current

    def update(self, result: Any, generation: Any = None, *, camera_pose: Any = None,
               now: Optional[float] = None) -> PanelState:
        clock, at, pose = _clock(now), float(getattr(result, "captured_at", math.nan)), _pose(camera_pose)
        age = math.inf if self._last_seen_at is None else max(0., clock-self._last_seen_at)
        if not math.isfinite(at) or not -.05 <= clock-at <= self.max_age_s or (self._last_frame_at is not None and at <= self._last_frame_at):
            if age > self.max_age_s:
                self.count = 0
            self.summary = PanelState(self.count >= self.required, False, False, self.count, self.missed, age, False)
            return self.summary
        reason = None
        shape = _shape(result)
        if self._generation is not None and generation is not None and generation != self._generation:
            reason = "GENERATION_CHANGED"
        elif self._image_shape is not None and shape != self._image_shape:
            reason = "IMAGE_SHAPE_CHANGED"
        elif _camera_changed(self._anchor_pose, pose):
            reason = "CAMERA_MOVED"
        elif age > self.max_age_s and self._last_seen_at is not None:
            reason = "PANEL_EXPIRED"
        box, rings, candidate = _box(result), _geometry(result), panel_candidate(result)
        if reason is None and candidate and self._anchor_box is not None:
            # Stable physical rims tolerate source-specific panel-box margins.
            same_rims = _layout_matches(self._anchor_rings, rings)
            if not same_rims and (not _box_consistent(self._anchor_box, box)
                                  or (min(len(self._anchor_rings), len(rings)) >= 4)):
                reason = "PANEL_LAYOUT_MOVED"
        if reason is not None:
            self.count = self.missed = 0
            self._last_seen_at = self._anchor_box = self._anchor_pose = None
            self._anchor_rings = []
        self._last_frame_at, self._image_shape = at, shape
        if generation is not None:
            self._generation = generation
        if candidate:
            if self._anchor_box is None:
                self._anchor_box, self._anchor_pose = box.copy(), pose
                self._anchor_rings = [r.copy() for r in rings]
            elif self._anchor_pose is None and pose is not None:
                self._anchor_pose = pose
            self.count += 1
            self.missed, self._last_seen_at = 0, at
            age = max(0., clock-at)
        else:
            self.missed += 1
            if self.missed > self.max_missed or age > self.max_age_s:
                self.count = 0
                self._anchor_box = self._anchor_pose = self._last_seen_at = None
                self._anchor_rings = []
                reason = reason or "PANEL_MISSING_LIMIT"
        self.summary = PanelState(self.count >= self.required, candidate, True, self.count, self.missed, age, candidate, reason)
        return self.summary


@dataclass(frozen=True)
class CachedLabel:
    track_id: int
    label: str
    confidence: float
    age_s: float
    valid_current: bool
    center_px: tuple[float, float]
    radius_px: float
    missed: int
    frames_ago: int = 0


@dataclass(frozen=True)
class LabelAssociation:
    track_id: int
    observation_index: int
    label: str
    confidence: float
    confirmed_observations: int
    confirmed_at: float
    captured_at: float
    generation: Any
    ring_xyr: tuple[float, float, float]
    frames_ago: int
    max_age_s: float
    max_frames: int


@dataclass(frozen=True)
class CurrentTrack:
    track_id: int
    observation_index: int
    label: Optional[str]
    confirmed_label: Optional[str]
    status: str
    support_count: int
    valid_current: bool


@dataclass(frozen=True)
class TemporalSummary:
    accepted_frame: bool
    current: tuple[CurrentTrack, ...]
    cached_labels: tuple[CachedLabel, ...]
    reset_reason: Optional[str] = None


@dataclass
class _Track:
    track_id: int
    ring: np.ndarray
    relative: np.ndarray
    anchor: np.ndarray
    last_at: float
    missed: int = 0
    confirmed_label: Optional[str] = None
    confirmed_score: float = 0.
    label_seen_at: float = -math.inf
    label_frame: int = 0
    label_pose: Optional[np.ndarray] = None
    confirmed_count: int = 0
    label_conflict: bool = False
    last_vote_label: Optional[str] = None
    streak: int = 0
    votes: deque = field(default_factory=deque)
    pixel_track_id: Optional[int] = None


def _label(observation: Any, min_score: float) -> tuple[Optional[str], float, str]:
    value=lambda key,default=None: observation.get(key,default) if isinstance(observation,dict) else getattr(observation,key,default)
    if "ERROR" in str(value("status", "")):
        return None, 0., "OCR_ERROR"
    raw, score = value("text"), value("confidence")
    if raw is None:
        return None, 0., "NO_TEXT"
    value = unicodedata.normalize("NFKC", str(raw)).strip().replace(" ", "")
    if not re.fullmatch(r"-?\d{1,3}", value):
        return None, 0., "NO_TEXT"
    if score is None or not math.isfinite(float(score)) or not min_score <= float(score) <= 1.:
        return None, 0., "LOW_CONF"
    return str(int(value)), float(score), "UNSTABLE"


class TemporalButtonTracker:
    def __init__(self, required: int = 3, max_age_s: float = 4., max_missed: int = 3, *,
                 min_score: float = .80, vote_window: int = 5, min_vote_fraction: float = .67,
                 min_ring_support: float = .85, max_age: Optional[float] = None,
                 memory_age_s: Optional[float] = None, memory_frames: int = 30):
        if max_age is not None:
            max_age_s = max_age  # Compatibility with the previous vision constructor.
        self.panel = PanelEvidence(required, max_age_s, max_missed)
        if int(vote_window) != vote_window or vote_window < required or not .5 < min_vote_fraction <= 1 or not 0 <= min_score <= 1 or not 0 <= min_ring_support <= 1:
            raise ValueError("投票窗口/多数比例/置信度参数无效")
        self.required, self.max_age_s, self.max_missed = int(required), float(max_age_s), int(max_missed)
        self.memory_age_s = self.max_age_s if memory_age_s is None else float(memory_age_s)
        if (not math.isfinite(self.memory_age_s) or not 0 < self.memory_age_s <= 30
                or int(memory_frames) != memory_frames or not 1 <= memory_frames <= 120):
            raise ValueError("标签缓存要求 0<memory_age_s<=30秒、1<=memory_frames<=120整数处理帧")
        self.memory_frames = int(memory_frames)
        self.min_score, self.vote_window, self.min_vote_fraction = min_score, int(vote_window), min_vote_fraction
        self.min_ring_support = min_ring_support
        self._next_id = 1
        self.reset()

    def reset(self) -> None:
        self.tracks: dict[int, _Track] = {}
        self.panel.reset()
        self.summary = TemporalSummary(False, (), ())
        self._frame_index = 0
        self._current_result = self._current_pose = self._current_generation = None

    def _cached(self, now: float, valid_ids: set[int]) -> tuple[CachedLabel, ...]:
        return tuple(CachedLabel(track.track_id, track.confirmed_label, track.confirmed_score,
                                 max(0., now-track.label_seen_at), track.track_id in valid_ids,
                                 tuple(float(v) for v in track.ring[:2]), float(track.ring[2]), track.missed,
                                 self._frame_index-track.label_frame)
                     for track in sorted(self.tracks.values(), key=lambda t: t.track_id)
                     if track.confirmed_label is not None and 0 <= now-track.label_seen_at <= self.memory_age_s
                     and self._frame_index-track.label_frame <= self.memory_frames)

    def cached_labels(self, now: Optional[float] = None) -> tuple[CachedLabel, ...]:
        return self._cached(_clock(now), set())

    def remembered_associations(self, result: Any, *, now: Optional[float] = None) -> tuple[LabelAssociation, ...]:
        """Bind confirmed label memory to a unique, strong rim in this frame.

        Missing/conflicting geometry, OCR failures, stale frames, excessive
        camera/anchor drift and conflicting labels cannot authorize this bridge.
        Current OCR fields and STABLE status are never synthesized or changed.
        """
        clock = _clock(now)
        if (result is not self._current_result or not self.summary.accepted_frame
                or not self.panel.stable or getattr(result, 'errors', ())
                or not 0 <= clock-float(result.captured_at) <= self.max_age_s):
            return ()
        memory = {item.track_id:item for item in self._cached(clock, set())}
        observations = result.buttons
        rings = [_ring(b, _shape(result)) for b in observations]
        matches = []
        for current in self.summary.current:
            track = self.tracks.get(current.track_id)
            saved = memory.get(current.track_id)
            index = current.observation_index
            metadata=getattr(result,'tracking',{})
            if metadata.get('workflow_evidence'):
                evidence=metadata.get('region_evidence',{}).get(str(index),{})
                if evidence.get('source')=='PIXEL_MATCHED_CACHE':
                    if not evidence.get('pixel_match'):continue
                    cached=metadata.get('display_label_cache',{}).get(str(index),{})
                    if saved is None or _label(cached,self.min_score)[0]!=saved.label:continue
                elif evidence.get('source')!='CURRENT_OCR':continue
            ring = rings[index]
            score = getattr(observations[index], 'confidence', None)
            if (track is None or saved is None or track.label_conflict or track.missed
                    or ring is None or ring[3] < self.min_ring_support
                    or 'ERROR' in str(observations[index].status)
                    or (score is not None and (not math.isfinite(float(score)) or not 0 <= float(score) <= 1))):
                continue
            if np.linalg.norm(ring[:2]-track.anchor[:2]) > max(4., .25*track.anchor[2]):
                continue
            if track.label_pose is not None and self._current_pose is not None:
                relative = track.label_pose[:3,:3].T @ self._current_pose[:3,:3]
                angle = math.acos(float(np.clip((np.trace(relative)-1)*.5, -1, 1)))
                if (np.linalg.norm(track.label_pose[:3,3]-self._current_pose[:3,3]) > .003
                        or angle > math.radians(2)):
                    continue
            # Greedy assignment alone is insufficient for remembered identity:
            # a duplicated/nested current candidate must not choose the target.
            nearby = sum(other is not None and .65 <= other[2]/ring[2] <= 1.5
                         and np.linalg.norm(other[:2]-ring[:2]) <= max(4., .35*ring[2]) for other in rings)
            if nearby != 1:
                continue
            matches.append(LabelAssociation(track.track_id, index, saved.label, saved.confidence,
                                            track.confirmed_count, track.label_seen_at, float(result.captured_at),
                                            self._current_generation, tuple(float(x) for x in track.ring[:3]),
                                            saved.frames_ago, self.memory_age_s, self.memory_frames))
        return tuple(matches)

    def update(self, result: Any, generation: Any = None, *, camera_pose: Any = None,
               now: Optional[float] = None) -> TemporalSummary:
        clock = _clock(now)
        panel = self.panel.update(result, generation, camera_pose=camera_pose, now=clock)
        if panel.reset_reason is not None:
            self.tracks.clear()
        self.tracks = {key: t for key, t in self.tracks.items()
                       if clock-t.last_at <= self.max_age_s and t.missed <= self.max_missed}
        observations = list(getattr(result, "buttons", ()))
        # A previous caller may have marked this very result STABLE. Always
        # invalidate its current status before duplicate/stale rejection.
        for observation in observations:
            _, _, initial = _label(observation, self.min_score)
            observation.status, observation.observations = initial, 0
            if hasattr(observation, "track_id"):
                observation.track_id = None
        if not panel.accepted_frame:
            self.summary = TemporalSummary(False, (), self._cached(clock, set()), panel.reset_reason)
            return self.summary
        self._frame_index += 1
        self._current_result, self._current_pose, self._current_generation = result, _pose(camera_pose), generation
        if not panel.valid_current:
            for track in self.tracks.values():
                track.missed += 1
            self.tracks = {key: t for key, t in self.tracks.items() if t.missed <= self.max_missed}
            self.summary = TemporalSummary(True, (), self._cached(clock, set()), panel.reset_reason)
            return self.summary
        at, box, shape = float(result.captured_at), _box(result), _shape(result)
        metadata=getattr(result,'tracking',{})
        sparse_ocr=bool(metadata.get('workflow_evidence'))
        geometries = {index: _ring(observation, shape) for index, observation in enumerate(observations)}
        geometries = {index: r for index, r in geometries.items() if r is not None
                      and box[0] <= r[0] <= box[2] and box[1] <= r[1] <= box[3]}
        relatives = {index: (r[:2]-box[:2])/(box[2:]-box[:2]) for index, r in geometries.items()}
        pairs = []
        for index, ring in geometries.items():
            for track_id, track in self.tracks.items():
                ratio = ring[2]/track.ring[2]
                distance = float(np.linalg.norm(ring[:2]-track.ring[:2]))
                relative = float(np.linalg.norm(relatives[index]-track.relative))
                anchor_drift = float(np.linalg.norm(ring[:2]-track.anchor[:2]))
                if (.65 <= ratio <= 1.5 and distance <= max(7., .9*max(ring[2], track.ring[2]))
                        and (relative <= .12 or distance <= max(3., .35*ring[2]))
                        and anchor_drift <= max(8., 1.1*ring[2])):
                    pairs.append((distance/max(4., ring[2]) + relative + .25*abs(math.log(ratio)), index, track_id))
        assigned, used = {}, set()
        for _, index, track_id in sorted(pairs):
            if index not in assigned and track_id not in used:
                assigned[index] = track_id
                used.add(track_id)
        for track_id, track in self.tracks.items():
            if track_id not in used:
                track.missed += 1
        current, valid_ids = [], set()
        for index, ring in geometries.items():
            track_id = assigned.get(index)
            if track_id is None:
                track_id, self._next_id = self._next_id, self._next_id+1
                self.tracks[track_id] = _Track(track_id, ring.copy(), relatives[index].copy(), ring.copy(), at)
            track = self.tracks[track_id]
            if sparse_ocr:
                evidence=metadata.get('region_evidence',{}).get(str(index),{})
                pixel_id=evidence.get('track_id')
                if pixel_id!=track.pixel_track_id:
                    track.votes.clear();track.streak=0;track.last_vote_label=None
                    track.confirmed_label=None;track.confirmed_score=0.;track.confirmed_count=0
                    track.label_seen_at=-math.inf;track.label_frame=0;track.label_pose=None
                    track.label_conflict=False;track.pixel_track_id=pixel_id
            track.ring, track.relative, track.last_at, track.missed = ring.copy(), relatives[index].copy(), at, 0
            vote_age=min(8.,self.memory_age_s) if sparse_ocr else self.max_age_s
            while track.votes and at-track.votes[0][0] > vote_age:
                track.votes.popleft()
            observation = observations[index]
            label, score, status = _label(observation, self.min_score)
            raw_label = _label(observation, 0.)[0]
            if sparse_ocr and evidence.get('source')!='CURRENT_OCR':
                label,score,status=None,0.,'NO_TEXT'
                cached=metadata.get('display_label_cache',{}).get(str(index),{})
                raw_label=_label(cached,0.)[0] if evidence.get('source')=='PIXEL_MATCHED_CACHE' else None
            if track.confirmed_label is not None and raw_label is not None and raw_label != track.confirmed_label:
                track.label_conflict = True
                if label is None:
                    track.streak, track.last_vote_label = 0, None
            eligible, support = False, 0
            if label is not None and ring[3] >= self.min_ring_support:
                track.streak = track.streak+1 if label == track.last_vote_label else 1
                track.last_vote_label = label
                track.votes.append((at, label, score))
                while len(track.votes) > self.vote_window:
                    track.votes.popleft()
                weights: dict[str, float] = {}
                for _, voted, confidence in track.votes:
                    weights[voted] = weights.get(voted, 0.) + confidence
                support = sum(voted == label for _, voted, _ in track.votes)
                winning = weights[label]/sum(weights.values()) >= self.min_vote_fraction
                eligible = winning and support >= self.required and track.streak >= self.required
                if eligible:
                    track.confirmed_label = label
                    track.confirmed_score = float(np.mean([confidence for _, voted, confidence in track.votes if voted == label]))
                    track.label_seen_at = at
                    track.label_frame, track.label_pose = self._frame_index, _pose(camera_pose)
                    track.confirmed_count, track.label_conflict = support, False
                if eligible and label == track.confirmed_label:
                    status = "STABLE"
                    valid_ids.add(track_id)
            elif label is not None:
                status = "UNSTABLE"
            observation.status, observation.observations = status, min(support, track.streak)
            observation.track_id = track_id
            current.append(CurrentTrack(track_id, index, label, track.confirmed_label, status,
                                        observation.observations, track_id in valid_ids))
        self.tracks = {key: t for key, t in self.tracks.items() if t.missed <= self.max_missed}
        self.summary = TemporalSummary(True, tuple(current), self._cached(clock, valid_ids), panel.reset_reason)
        return self.summary


# Drop-in name for the previous entry's update(result)/reset() usage.
StabilityTracker = TemporalButtonTracker
