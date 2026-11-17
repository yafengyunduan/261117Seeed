"""Independent cabin-floor display reader; no camera or robot access here.

Locate a horizontal luminous display inside a dark surround, recognize only
its right-hand text, and confirm several independent OCR observations. This
branch never reads floor buttons or infers floor labels from their layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import time
import unicodedata

import cv2
import numpy as np

from .ocr import build_ocr


def order_quad(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("显示区四角必须有限")
    # Left and right pairs also work for an upright, mildly rolled panel.
    left, right = np.array_split(points[np.argsort(points[:, 0])], 2)
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    quad = np.float32([tl, tr, br, bl])
    if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 16:
        raise ValueError("显示区四角无效")
    return quad


def quad_box(quad, shape):
    h, w = shape[:2]
    lo = np.floor(np.min(quad, axis=0)).astype(int)
    hi = np.ceil(np.max(quad, axis=0)).astype(int) + 1
    return (max(0, int(lo[0])), max(0, int(lo[1])),
            min(w, int(hi[0])), min(h, int(hi[1])))


def rectify(image, quad):
    """Return bounded screen and exact inverse perspective map to source RGB."""
    quad = order_quad(quad)
    width = float((np.linalg.norm(quad[1]-quad[0]) + np.linalg.norm(quad[2]-quad[3]))/2)
    height = float((np.linalg.norm(quad[3]-quad[0]) + np.linalg.norm(quad[2]-quad[1]))/2)
    ratio = width/max(height, 1.)
    if not 1.45 < ratio < 6. or height < 8:
        raise ValueError("显示区宽高比或尺寸不适合此分支")
    ww = min(320, max(80, round(width)))
    hh = max(16, round(ww/ratio))
    target = np.float32([[0, 0], [ww-1, 0], [ww-1, hh-1], [0, hh-1]])
    matrix = cv2.getPerspectiveTransform(quad, target)
    inverse = cv2.getPerspectiveTransform(target, quad)
    return cv2.warpPerspective(image, matrix, (ww, hh)), inverse


def text_region(screen):
    """Measured bright glyphs in the screen right side; ignore left arrows.

    No letter-to-digit substitutions: OCR V, S, B etc. are never silently
    converted into numbers. The supported display layout has an arrow at left.
    """
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x0, x1 = round(.43*w), round(.97*w)
    y0, y1 = round(.08*h), round(.92*h)
    region = gray[y0:y1, x0:x1]
    median, high = np.percentile(region, [50, 99])
    if high-median < 18:
        return None
    otsu, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold = max(float(otsu), float(median)+max(10., .25*float(high-median)))
    mask = np.uint8(region > threshold)*255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    selected = []
    for index in range(1, count):
        x, y, ww, hh, area = stats[index]
        if (.25*h <= hh <= .85*h and .075 <= ww/hh <= 1.5
                and area >= max(8., .015*hh*hh) and x > 0
                and y > 0 and y+hh < region.shape[0]):
            selected.append((index, int(x), int(y), int(ww), int(hh)))
    if not selected or len(selected) > 4:
        return None
    selected.sort(key=lambda s: s[1])
    heights = [s[4] for s in selected]
    if min(heights)/max(heights) < .55:
        return None
    for a, b in zip(selected, selected[1:]):
        if b[1]-(a[1]+a[3]) > .9*max(heights):
            return None
    bx0 = min(s[1] for s in selected)+x0
    by0 = min(s[2] for s in selected)+y0
    bx1 = max(s[1]+s[3] for s in selected)+x0
    by1 = max(s[2]+s[4] for s in selected)+y0
    # Include an actual minus stroke left of the measured digits.
    for index in range(1, count):
        xx, yy, ww, hh, area = stats[index]
        if (.18*max(heights) < ww < .8*max(heights) and hh < .25*max(heights)
                and bx0-.9*max(heights) < xx+x0 < bx0
                and abs(yy+y0+hh/2-(by0+by1)/2) < .25*max(heights) and area > 5):
            bx0 = min(bx0, int(xx+x0))
    margin = max(3, round(.12*(by1-by0)))
    box = (max(x0, bx0-margin), max(0, by0-margin), min(w, bx1+margin), min(h, by1+margin))
    a, b, c, d = box
    patch = screen[b:d, a:c].copy()
    if patch.size == 0:
        return None
    glyph = cv2.resize(mask[min(s[2] for s in selected):max(s[2]+s[4] for s in selected),
                            max(0, bx0-x0):min(mask.shape[1], bx1-x0)], (64, 64), interpolation=cv2.INTER_AREA)
    return patch, box, glyph.astype(np.float32)/255.


def _screen_check(image, quad, *, need_text=False):
    h, w = image.shape[:2]
    if np.any(quad[:, 0] < 2) or np.any(quad[:, 1] < 2) or np.any(quad[:, 0] >= w-2) or np.any(quad[:, 1] >= h-2):
        return None
    try:
        screen, inverse = rectify(image, quad)
    except (ValueError, cv2.error):
        return None
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    if not 8 <= median <= 120:
        return None
    blue, green, red = np.median(screen.reshape(-1, 3), axis=0)
    # A red registration/name plate below the black panel is not an LCD.
    # Check the dominant background, so red LED strokes on black still work.
    if red-blue > 20 and red-green > 20:
        return None
    # Compare background with the dark panel immediately above and below LCD.
    sh, sw = gray.shape
    screen_points = np.float32([[[x, y] for x in np.linspace(.12*(sw-1), .88*(sw-1), 12)]
                               for y in (-.22*sh, 1.22*sh)])
    mapped = cv2.perspectiveTransform(screen_points.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    xx, yy = np.rint(mapped[:, 0]).astype(int), np.rint(mapped[:, 1]).astype(int)
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    if valid.sum() < 12:
        return None
    outside = cv2.cvtColor(image[yy[valid], xx[valid]][None], cv2.COLOR_BGR2GRAY)[0]
    dark_fraction = float(np.mean(outside < min(100., median+12)))
    contrast = median-float(np.median(outside))
    if dark_fraction < .70 or contrast < 4.:
        return None
    text = text_region(screen)
    if need_text and text is None:
        return None
    return screen, inverse, text, dict(background_median=median, surround_contrast=contrast,
                                       dark_surround_fraction=dark_fraction)


def find_display(image, max_side=960):
    """Bounded geometry-only search. Image names and floor labels never enter."""
    h, w = image.shape[:2]
    scale = min(1., max_side/max(h, w))
    ww, hh = max(1, round(w*scale)), max(1, round(h*scale))
    work = image
    # Integer pyramid steps avoid a costly large non-integer area reduction on
    # 12MP reference photos; the final mapping still uses original RGB sizes.
    while max(work.shape[:2]) > 2*max_side:
        work = cv2.pyrDown(work)
    work = cv2.resize(work, (ww, hh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.GaussianBlur(cv2.createCLAHE(2., (8, 8)).apply(gray), (3, 3), 0)
    choices = []
    for low, high in ((20, 45), (40, 90)):
        edges = cv2.morphologyEx(cv2.Canny(enhanced, low, high), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            # Tiny manufacturer/capacity labels are not the large floor LCD.
            # A smaller real display can still be specified explicitly by ROI.
            if bh < .03*hh or not 1.5 < bw/bh < 5.5 or not .0015*ww*hh < bw*bh < .15*ww*hh:
                continue
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            # A screen border may have a small gap. Its connected arc still
            # supports a rectangle even when the raw contour encloses no area.
            if not .0015*ww*hh < area < .12*ww*hh or area/(bw*bh) < .60:
                continue
            poly = cv2.approxPolyDP(hull, .025*cv2.arcLength(hull, True), True)
            points = poly[:, 0] if len(poly) == 4 else cv2.boxPoints(cv2.minAreaRect(hull))
            try:
                local = order_quad(points)
            except ValueError:
                continue
            checked = _screen_check(work, local, need_text=True)
            if checked is None:
                continue
            # Collapse duplicate contours of the same LCD, never choose among
            # distinct screens by OCR confidence or expected floor number.
            duplicate = next((i for i, q in enumerate(choices)
                              if np.linalg.norm(np.mean(q, axis=0)-np.mean(local, axis=0)) < .3*max(bw, bh)), None)
            if duplicate is None:
                choices.append(local)
            elif cv2.contourArea(local) > cv2.contourArea(choices[duplicate]):
                choices[duplicate] = local
    if len(choices) != 1:
        return None, 'MULTIPLE_DISPLAYS' if choices else 'NO_DISPLAY'
    quad = choices[0]/np.float32([ww/w, hh/h])
    return quad, 'DISPLAY_FOUND'


def normalize_floor(text):
    value = unicodedata.normalize('NFKC', str(text)).strip().upper().replace('−', '-').replace('–', '-')
    if re.fullmatch(r'-?\d{1,3}', value):
        return str(int(value))
    if re.fullmatch(r'B[1-9]\d?', value):
        return value
    return None


class DisplayOCR:
    def __init__(self, *, device='auto', gpu_index=0, threads=2):
        from rapidocr.ch_ppocr_rec import TextRecInput
        self.input_type = TextRecInput
        self.engine, self.info = build_ocr(threads=threads, device=device, gpu_index=gpu_index)
        self.calls = 0

    def read(self, patch):
        # One small recognition crop; a second polarity is only a retry after
        # weak/non-numeric output. No full-frame OCR or text detector is loaded.
        tick = time.perf_counter()
        if self.info['actual_device'].startswith('cuda'):
            import torch
            torch.cuda.synchronize(self.info['gpu_index'])
        value = None
        for candidate in (patch, 255-patch):
            output = self.engine.text_rec(self.input_type(img=[candidate]))
            self.calls += 1
            if output.txts is None or output.scores is None or len(output.txts) != 1 or len(output.scores) != 1:
                raise RuntimeError('OCR 输出数量不匹配')
            found = (str(output.txts[0]), float(output.scores[0]))
            if not math.isfinite(found[1]) or not 0 <= found[1] <= 1:
                raise RuntimeError('OCR 置信度无效')
            if value is None or normalize_floor(found[0]) is not None and (normalize_floor(value[0]) is None or found[1] > value[1]):
                value = found
            if normalize_floor(value[0]) is not None and value[1] >= .80:
                break
        if self.info['actual_device'].startswith('cuda'):
            torch.cuda.synchronize(self.info['gpu_index'])
        return value, (time.perf_counter()-tick)*1000


@dataclass
class DisplayResult:
    captured_at: float | None
    shape: tuple
    state: str = 'NO_DISPLAY'
    reason: str = 'NO_DISPLAY'
    observed_floor: str | None = None
    current_floor: str | None = None
    last_confirmed_floor: str | None = None
    last_confirmed_age_s: float | None = None
    confidence: float | None = None
    raw_text: str | None = None
    display_quad: np.ndarray | None = None
    text_quad: np.ndarray | None = None
    crop: np.ndarray | None = None
    source: str = 'NONE'
    geometry_source: str = 'NONE'
    independent_votes: int = 0
    ocr_age_s: float | None = None
    backend: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    generation: int = 0

    def to_dict(self):
        return dict(branch='CABIN_FLOOR_DISPLAY', captured_at=self.captured_at, image_shape_hw=list(self.shape),
                    state=self.state, reason=self.reason, observed_floor=self.observed_floor,
                    current_floor=self.current_floor, confidence=self.confidence,
                    raw_text=self.raw_text,
                    last_confirmed_floor=self.last_confirmed_floor, last_confirmed_age_s=self.last_confirmed_age_s,
                    display_polygon_px=None if self.display_quad is None else self.display_quad.tolist(),
                    text_polygon_px=None if self.text_quad is None else self.text_quad.tolist(),
                    text_center_px=None if self.text_quad is None else np.mean(self.text_quad, axis=0).tolist(),
                    source=self.source, geometry_source=self.geometry_source,
                    independent_ocr_votes=self.independent_votes, ocr_age_s=self.ocr_age_s,
                    ocr_backend=self.backend, timings_ms=self.timings_ms, errors=self.errors,
                    generation=self.generation, actions_enabled=False)


class FloorDisplayReader:
    def __init__(self, ocr, *, min_confidence=.80, required=3, interval_s=.20, max_age_s=.8,
                 refresh_s=1.5, max_side=960, roi=None):
        if not .5 <= min_confidence <= 1 or not 2 <= required <= 20 or int(required) != required:
            raise ValueError('置信度需0.5..1；连续OCR次数需2..20整数')
        if not all(math.isfinite(v) and v > 0 for v in (interval_s, max_age_s, refresh_s)) or interval_s >= max_age_s:
            raise ValueError('时间参数需有限正数，OCR间隔须小于缓存有效期')
        if not 480 <= max_side <= 1600:
            raise ValueError('获取分辨率长边需480..1600')
        if roi is not None and (len(roi) != 4 or not np.isfinite(roi).all() or roi[2] <= roi[0] or roi[3] <= roi[1]):
            raise ValueError('ROI需x1 y1 x2 y2，范围有效')
        self.ocr, self.min_confidence, self.required = ocr, min_confidence, required
        self.interval_s, self.max_age_s, self.refresh_s, self.max_side = interval_s, max_age_s, refresh_s, max_side
        self.user_roi = roi
        self.reset()

    def reset(self):
        self.quad = self.previous = self.track_box = None
        self.last_at = self.last_search = self.last_ocr = -math.inf
        self.next_ocr = -math.inf
        self.signature = None
        self.label = self.confidence = None
        self.votes = 0
        self.content_pending = False
        self.generation = self.shape = None
        self.next_search = -math.inf
        self.confirmed = None
        self.confirmed_at = -math.inf

    def _clear_text(self):
        self.label = self.confidence = self.signature = None
        self.votes = 0
        self.last_ocr = -math.inf
        self.content_pending = False

    def _seed(self, image, quad):
        h, w = image.shape[:2]
        lo, hi = np.min(quad, axis=0), np.max(quad, axis=0)
        pad = max(12., .35*(hi[0]-lo[0]))
        box = (max(0, int(lo[0]-pad)), max(0, int(lo[1]-pad)), min(w, int(hi[0]+pad)), min(h, int(hi[1]+pad)))
        x0, y0, x1, y1 = box
        patch = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        scale = min(1., 480/max(patch.shape))
        self.previous = cv2.resize(patch, (max(1, round(patch.shape[1]*scale)), max(1, round(patch.shape[0]*scale))))
        self.track_box = box
        self.quad = quad

    def _track(self, image):
        x0, y0, x1, y1 = self.track_box
        raw = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        current = cv2.resize(raw, self.previous.shape[::-1])
        sx, sy = current.shape[1]/raw.shape[1], current.shape[0]/raw.shape[0]
        mask = np.full(current.shape, 255, np.uint8)
        inner = ((self.quad-np.array([x0, y0]))*np.array([sx, sy])).astype(np.int32)
        cv2.fillConvexPoly(mask, inner, 0)  # Changing digits must not drive camera motion estimates.
        points = cv2.goodFeaturesToTrack(self.previous, 70, .012, 3, mask=mask, blockSize=5)
        if points is not None and len(points) >= 6:
            following, status, error = cv2.calcOpticalFlowPyrLK(self.previous, current, points, None, winSize=(21, 21), maxLevel=2)
            if following is not None:
                back, good, _ = cv2.calcOpticalFlowPyrLK(current, self.previous, following, None, winSize=(21, 21), maxLevel=2)
                if back is not None:
                    valid = (status[:, 0] > 0) & (good[:, 0] > 0) & (np.linalg.norm(back[:, 0]-points[:, 0], axis=1) < 1.2)
                    if error is not None:
                        valid &= error[:, 0] < 30
                    if valid.sum() >= 6:
                        matrix, inliers = cv2.estimateAffinePartial2D(points[valid, 0], following[valid, 0], method=cv2.RANSAC, ransacReprojThreshold=2.)
                        if matrix is not None and inliers is not None and inliers.mean() >= .7:
                            scale = math.hypot(matrix[0, 0], matrix[1, 0])
                            angle = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
                            if .90 <= scale <= 1.10 and angle <= 12:
                                local = (self.quad-[x0, y0])*[sx, sy]
                                moved = cv2.transform(local.astype(np.float32)[None], matrix)[0]/[sx, sy]+[x0, y0]
                                moved = moved.astype(np.float32)
                                checked = _screen_check(image, moved)
                                if checked is not None:
                                    return moved, checked, 'LOCAL_FLOW'
        checked = _screen_check(image, self.quad)
        return (self.quad, checked, 'LOCAL_SCREEN_CHECK') if checked is not None else (None, None, 'NONE')

    def process(self, image, captured_at=None, *, offline=False, generation=0):
        tick = time.perf_counter()
        try:
            at = tick if captured_at is None else float(captured_at)
        except (TypeError, ValueError):
            at = math.nan
        shape = image.shape[:2] if isinstance(image, np.ndarray) and image.ndim >= 2 else (0, 0)
        result = DisplayResult(at if math.isfinite(at) else None, shape, backend=dict(self.ocr.info), generation=generation)
        if not math.isfinite(at):
            self.reset(); result.state = result.reason = 'INVALID_TIMESTAMP'; return result
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(shape) < 24:
            self.reset(); result.state = result.reason = 'INVALID_FRAME'; return result
        if offline or self.generation != generation or self.shape != shape:
            self.reset(); self.generation = generation; self.shape = shape
        if at <= self.last_at:
            self._clear_text(); result.state = result.reason = 'STALE_FRAME'; return result
        if at-self.last_at > self.max_age_s:
            self._clear_text()
        self.last_at = at
        quad = checked = None
        start = time.perf_counter()
        if self.user_roi is not None:
            x0, y0, x1, y1 = self.user_roi
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            checked = _screen_check(image, quad)
            result.geometry_source = 'MANUAL_ROI'
        else:
            if self.quad is not None:
                quad, checked, result.geometry_source = self._track(image)
            if (quad is None and at >= self.next_search) or quad is not None and at-self.last_search >= self.refresh_s:
                found, reason = find_display(image, self.max_side)
                self.last_search = at
                self.next_search = at+.3
                if found is not None:
                    # A changed physical screen cannot inherit a confirmed floor.
                    if self.quad is not None and np.linalg.norm(np.mean(found, axis=0)-np.mean(self.quad, axis=0)) > .2*np.linalg.norm(self.quad[1]-self.quad[0]):
                        self._clear_text()
                    quad, checked = found, _screen_check(image, found)
                    result.geometry_source = 'FULL_RECTANGLE'
                elif reason == 'MULTIPLE_DISPLAYS':
                    quad = checked = None
                    self.quad = self.previous = self.track_box = None
                    result.reason = reason
        result.timings_ms['geometry'] = (time.perf_counter()-start)*1000
        if quad is None or checked is None:
            self._clear_text()
            result.reason = result.reason if result.reason == 'MULTIPLE_DISPLAYS' else 'NO_DISPLAY'
            result.state = result.reason
        else:
            result.display_quad = quad
            screen, inverse, text, features = checked
            result.crop = screen
            self._seed(image, quad)
            if text is None:
                self._clear_text(); result.state = result.reason = 'NO_TEXT'
            else:
                patch, box, signature = text
                x0, y0, x1, y1 = box
                result.text_quad = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), inverse)[0]
                changed = self.signature is not None and float(np.mean(np.abs(signature-self.signature))) > .10
                if changed:
                    # Hold output pending current OCR; preserve the previous
                    # consensus only if that new OCR confirms the same text.
                    self.content_pending = True
                result.state = result.reason = 'WAITING_OCR'
                due = at >= self.next_ocr
                if due:
                    self.next_ocr = at+self.interval_s
                    try:
                        (raw, score), ocr_ms = self.ocr.read(patch)
                        result.raw_text = raw
                        result.timings_ms['ocr'] = ocr_ms
                        floor = normalize_floor(raw)
                        if floor is None or score < self.min_confidence:
                            self._clear_text()
                            result.state = result.reason = 'LOW_CONFIDENCE' if floor is not None else 'NON_FLOOR_TEXT'
                        else:
                            self.votes = min(self.required, self.votes+1) if floor == self.label else 1
                            self.label, self.confidence, self.last_ocr = floor, score, at
                            self.signature = signature
                            self.content_pending = False
                            result.source = 'CURRENT_OCR'
                    except Exception as exc:
                        self._clear_text(); result.state = result.reason = 'OCR_ERROR'
                        result.errors.append(f'{type(exc).__name__}: {exc}')
                elif self.label is not None and not self.content_pending:
                    result.source = 'PIXEL_VERIFIED_CACHE'
                if self.label is not None and not self.content_pending and at-self.last_ocr <= self.max_age_s:
                    result.observed_floor, result.confidence = self.label, self.confidence
                    result.independent_votes = self.votes
                    result.ocr_age_s = max(0., at-self.last_ocr)
                    result.state = 'SINGLE_SHOT' if offline else 'STABLE' if self.votes >= self.required else 'UNSTABLE'
                    result.reason = 'DISPLAY_TEXT_RECOGNIZED'
                    if result.state == 'STABLE':
                        result.current_floor = self.label
                        self.confirmed, self.confirmed_at = self.label, at
        if self.confirmed is not None and 0 <= at-self.confirmed_at <= 2.:
            result.last_confirmed_floor = self.confirmed
            result.last_confirmed_age_s = at-self.confirmed_at
        result.timings_ms['total'] = (time.perf_counter()-tick)*1000
        return result


def draw_result(image, result, max_side=1280):
    h, w = image.shape[:2]
    scale = min(1., max_side/max(h, w))
    display = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
    sx, sy = display.shape[1]/w, display.shape[0]/h
    def put(text, xy, color=(255, 255, 255), size=.55):
        cv2.putText(display, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    if result.display_quad is not None:
        points = np.rint(result.display_quad*[sx, sy]).astype(np.int32)
        cv2.polylines(display, [points], True, (0, 220, 255), 2, cv2.LINE_AA)
    if result.text_quad is not None:
        points = np.rint(result.text_quad*[sx, sy]).astype(np.int32)
        color = (50, 230, 50) if result.state == 'STABLE' else (255, 220, 0)
        cv2.polylines(display, [points], True, color, 2, cv2.LINE_AA)
        put(f'FLOOR {result.observed_floor or "?"}', tuple(points[0]-[0, 7]), color, .65)
    floor = result.current_floor or result.observed_floor or '?'
    score = '--' if result.confidence is None else f'{result.confidence:.5f}'
    put(f'CABIN DISPLAY | floor={floor} score={score} {result.state}', (8, 24))
    put(f'{result.backend.get("actual_backend", "?")} | {result.source} | {result.timings_ms.get("total", 0):.1f}ms', (8, 48), size=.48)
    if result.current_floor is None and result.last_confirmed_floor is not None:
        put(f'last confirmed={result.last_confirmed_floor} ({result.last_confirmed_age_s:.2f}s ago)', (8, 69), (0, 220, 255), .48)
    if result.crop is not None:
        crop = result.crop
        cw = min(300, max(100, display.shape[1]//3))
        ch = round(cw*crop.shape[0]/crop.shape[1])
        inset = cv2.resize(crop, (cw, ch))
        if ch+90 < display.shape[0]:
            display[70:70+ch, display.shape[1]-cw-8:display.shape[1]-8] = inset
    put('NO ROBOT | Q/Esc=quit R=reset', (8, display.shape[0]-12), size=.5)
    return display
