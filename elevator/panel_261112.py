"""Shared panel recognition (2026-11-12): ROI recovery, OCR retry, verified display cache.

No camera, robot, or executable entrypoint. Both main workflows and the
independent preview use this module. Models initialize only on construction.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import replace
import math
import time
import cv2
import numpy as np
from . import perception as vision

# Fixed feature checks, not OCR confidence or probabilities.
THRESHOLDS = (140, 160, 175, 190, 200, 210, 225, 235)

def triangle_candidates(mask, radius):
    # Native-resolution contours retain the component topology.
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        bx,by,bw,bh=cv2.boundingRect(contour)
        if not (.045*radius*radius < area < .40*radius*radius and .48*radius < bh < 1.20*radius and .27 < bw/bh < .88):continue
        fit_area, tri=cv2.minEnclosingTriangle(contour)
        if tri is None or fit_area<1:continue
        fill=area/fit_area
        if fill<.72:continue
        vertices=tri.reshape(3,2)
        # In these filled isosceles door arrows, the base is the longest side.
        edges=[(vertices[i],vertices[(i+1)%3],vertices[(i+2)%3]) for i in range(3)]
        p,q,tip=max(edges,key=lambda e:float(np.linalg.norm(e[0]-e[1])))
        base=(p+q)/2; bvec=q-p; length=float(np.linalg.norm(bvec)); unit=bvec/length
        offset=tip-base; parallel=float(np.dot(offset,unit)); normal=offset-parallel*unit
        width=float(np.linalg.norm(normal))
        if length < .48*radius or abs(parallel)>.3*length or not .25*length<width<.82*length:continue
        # Limit this prototype to nearly horizontal door-arrow pairs.
        if abs(float(bvec[0]))>.5*abs(float(bvec[1])):continue
        shape=fill*(1-min(1,abs(parallel)/(.4*length))*.3)
        out.append(dict(centroid=[float(np.mean(vertices[:,0])),float(np.mean(vertices[:,1]))],direction=1 if normal[0]>0 else -1,score=shape,area=area,height=length,width=width,base_vector=bvec.tolist(),vertices=vertices.tolist(),fill=fill))
    return out

def has_separator(mask,radius,center,base_vector):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    base=np.asarray(base_vector);base=base/np.linalg.norm(base)
    for contour in contours:
        rect=cv2.minAreaRect(contour);(cx,cy),(ww,hh),_=rect
        short,long=min(ww,hh),max(ww,hh)
        if not (.52*radius<long<1.25*radius and .012*radius<short<.22*radius and short/long<.22):continue
        if np.linalg.norm(np.asarray([cx,cy])-center)>.22*radius:continue
        if cv2.contourArea(contour)/max(1,ww*hh)<.45:continue
        points=cv2.boxPoints(rect);edges=[points[1]-points[0],points[2]-points[1]]
        vec=max(edges,key=np.linalg.norm);vec=vec/np.linalg.norm(vec)
        if abs(float(np.dot(vec,base)))>.95:return True
    return False

def classify(image,ring,polarities=('bright','dark')):
    if (image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or not np.isfinite([ring.x,ring.y,ring.radius]).all()
            or ring.radius < 8 or not 0 <= ring.x < image.shape[1] or not 0 <= ring.y < image.shape[0]):
        return dict(label='UNKNOWN',score=0.,scores={'OPEN':0.,'CLOSE':0.},best=None,thresholds={}),[]
    r=ring.radius;x1=max(0,math.floor(ring.x-.78*r));y1=max(0,math.floor(ring.y-.82*r));x2=min(image.shape[1],math.ceil(ring.x+.78*r));y2=min(image.shape[0],math.ceil(ring.y+.82*r))
    patch=image[y1:y2,x1:x2];gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    yy,xx=np.indices(gray.shape);central=((xx+x1-ring.x)**2+(yy+y1-ring.y)**2<(.82*r)**2)
    results={'OPEN':0.,'CLOSE':0.};best=None;all_triangles=[]
    used_thresholds={}
    for polarity in polarities:
      work=gray if polarity=='bright' else 255-gray
      pp=work[central]
      values=np.percentile(pp,[50,70,78,84,89,93,99]);lo,hi=float(values[0]),float(values[-1])
      thresholds=sorted(set(int(max(value,lo+a*(hi-lo))) for value,a in zip(values[1:-1],[.4,.5,.6,.7,.8])))
      used_thresholds[polarity]=thresholds
      for threshold in thresholds:
        mask=np.uint8((work>=threshold)&central)*255
        tris=triangle_candidates(mask,r)
        for tri in tris:
            tri['centroid'][0]+=x1;tri['centroid'][1]+=y1
            tri['vertices']=(np.asarray(tri['vertices'])+[x1,y1]).tolist()
        all_triangles.append((threshold,tris))
        left=[t for t in tris if -.62*r < t['centroid'][0]-ring.x < -.12*r and -.38*r<t['centroid'][1]-ring.y<.38*r]
        right=[t for t in tris if .12*r < t['centroid'][0]-ring.x < .62*r and -.38*r<t['centroid'][1]-ring.y<.38*r]
        for l in left:
            for rr in right:
                if l['direction']==rr['direction']:continue
                axis=np.subtract(rr['centroid'],l['centroid']);axis_length=float(np.linalg.norm(axis));axis_unit=axis/axis_length
                h_ratio=min(l['height'],rr['height'])/max(l['height'],rr['height'])
                area_ratio=min(l['area'],rr['area'])/max(l['area'],rr['area'])
                symmetry=abs((l['centroid'][0]+rr['centroid'][0])/2-ring.x)/r
                if abs(float(axis[1]))>.45*abs(float(axis[0])) or h_ratio<.68 or area_ratio<.55 or symmetry>.18:continue
                alignment=max(abs(float(np.dot(np.asarray(t['base_vector'])/t['height'],axis_unit))) for t in (l,rr))
                if alignment>.20:continue
                if not has_separator(mask,r,[(l['centroid'][i]+rr['centroid'][i])/2-[x1,y1][i] for i in range(2)],l['base_vector']):continue
                label='OPEN' if l['direction']<0 else 'CLOSE'
                score=math.sqrt(l['score']*rr['score'])*h_ratio**.3*area_ratio**.15*(1-alignment)*(1-symmetry)
                results[label]=max(results[label],score)
                if best is None or score>best['score']:best=dict(label=label,score=score,threshold=threshold,polarity=polarity,triangles=[l,rr])
    label=best['label'] if best and best['score']>=.58 and abs(results['OPEN']-results['CLOSE'])>=.12 else 'UNKNOWN'
    return dict(label=label,score=best['score'] if best else 0.,scores=results,best=best,thresholds=used_thresholds),all_triangles

def classify_door(image,ring):
    """Classify one ring; no model loading, OCR, camera, or robot access."""
    # Small fixed-size shape masks avoid contour/percentile work proportional
    # to phone-photo resolution. The same checks run at radius <=45 pixels.
    result=None
    if (image is not None and image.ndim==3 and image.shape[2]==3 and image.dtype==np.uint8
            and np.isfinite([ring.x,ring.y,ring.radius]).all() and ring.radius>45
            and 0<=ring.x<image.shape[1] and 0<=ring.y<image.shape[0]):
        x0=max(0,math.floor(ring.x-.92*ring.radius));y0=max(0,math.floor(ring.y-.92*ring.radius))
        x1=min(image.shape[1],math.ceil(ring.x+.92*ring.radius));y1=min(image.shape[0],math.ceil(ring.y+.92*ring.radius))
        patch=image[y0:y1,x0:x1];scale=45/ring.radius
        ww,hh=max(1,round(patch.shape[1]*scale)),max(1,round(patch.shape[0]*scale))
        sx,sy=ww/patch.shape[1],hh/patch.shape[0]
        resized=cv2.resize(patch,(ww,hh),interpolation=cv2.INTER_AREA)
        local=vision.Ring((ring.x-x0)*sx,(ring.y-y0)*sy,ring.radius*math.sqrt(sx*sy),1.)
        result=classify(resized,local)[0]
        if result['best'] is not None:
            for triangle in result['best']['triangles']:
                triangle['centroid']=(np.asarray(triangle['centroid'])/[sx,sy]+[x0,y0]).tolist()
                triangle['vertices']=(np.asarray(triangle['vertices'])/[sx,sy]+[x0,y0]).tolist()
    if result is None:result=classify(image,ring)[0]
    result['center_px']=([round(ring.x),round(ring.y)]
                         if np.isfinite([ring.x,ring.y]).all() else None)
    if result['best'] is not None:
        points=np.asarray([v for t in result['best']['triangles'] for v in t['vertices']])
        result['glyph_bbox_xyxy']=[int(math.floor(points[:,0].min())),int(math.floor(points[:,1].min())),int(math.ceil(points[:,0].max())),int(math.ceil(points[:,1].max()))]
    else:
        result['glyph_bbox_xyxy']=None
    return result

def glyph_shape_evidence(image, center, radius):
    x, y = center
    r = float(radius)
    if (image is None or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8
            or not np.isfinite([x,y,r]).all() or r<8):
        return dict(status='UNKNOWN',reason='INVALID_OR_SMALL_PATCH',shape_candidates=[],
                    probability=None,executable=False)
    x0, y0 = round(x-.70*r), round(y-.80*r)
    x1, y1 = round(x+.70*r), round(y+.45*r)
    if r < 8 or x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return {"status": "UNKNOWN", "reason": "CLIPPED_PATCH"}
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    hits = Counter()
    per_threshold = []
    for threshold in THRESHOLDS:
        mask = np.uint8(gray >= threshold)*255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        components = []
        for bx, by, bw, bh, area in stats[1:].tolist():
            touching = (bx <= 1 or by <= 1 or bx+bw >= gray.shape[1]-1
                        or by+bh >= gray.shape[0]-1)
            components.append(dict(cx=(bx+bw/2+x0-x)/r,
                cy=(by+bh/2+y0-y)/r, w=bw/r, h=bh/r, area=area/r**2,
                touching=touching))
        verticals = [c for c in components if not c["touching"]
                     and -.12 <= c["cx"] <= .30
                     and .60 <= c["h"] <= 1.10
                     and .025 <= c["w"] <= .18
                     and c["h"]/c["w"] >= 4
                     and c["area"] >= .02]
        clean_vertical = False
        for vertical in verticals:
            other_tall = [c for c in components if c is not vertical
                          and not c["touching"] and c["h"] > .45
                          and c["area"] > .035]
            if not other_tall:
                clean_vertical = True
        if clean_vertical:
            hits["isolated_vertical_stroke"] += 1

        # A short bright minus can connect to a radial reflection; horizontal
        # opening checks the actual line pixels instead of the entire blob.
        kernel = np.ones((1, max(3, round(.22*r))), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        _, _, horizontal_stats, _ = cv2.connectedComponentsWithStats(opened)
        minus = False
        for bx, by, bw, bh, area in horizontal_stats[1:].tolist():
            cx, cy = (bx+bw/2+x0-x)/r, (by+bh/2+y0-y)/r
            if (-.60 < cx < -.08 and -.25 < cy < .10
                    and .22 <= bw/r < .60 and bh/r < .22
                    and area/r**2 >= .012):
                minus = True
        if minus:
            hits["left_horizontal_stroke"] += 1

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
        holes = []
        if hierarchy is not None:
            for contour, info in zip(contours, hierarchy[0]):
                if info[3] < 0:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)/r**2
                if .16 <= bw/r <= .55 and .16 <= bh/r <= .55 and area >= .025:
                    holes.append(dict(cx=(bx+bw/2+x0-x)/r,
                        cy=(by+bh/2+y0-y)/r, area=area, parent=int(info[3])))
        stacked_holes = any(a["parent"] == b["parent"]
                            and abs(a["cx"]-b["cx"]) <= .16
                            and .22 <= abs(a["cy"]-b["cy"]) <= .70
                            and .30 <= a["area"]/b["area"] <= 3.33
                            for i, a in enumerate(holes) for b in holes[i+1:])
        if stacked_holes:
            hits["two_stacked_holes"] += 1
        per_threshold.append(dict(threshold=threshold,
            isolated_vertical=clean_vertical, left_horizontal=minus,
            two_stacked_holes=stacked_holes))

    # These remain visual hypotheses. I, door icons, B, damage and other fonts
    # are counterexamples: additional OCR and region checks are required.
    shape_candidates = []
    if hits["isolated_vertical_stroke"] >= 2:
        shape_candidates.append("-1" if hits["left_horizontal_stroke"] >= 3 else "1")
    if hits["two_stacked_holes"] >= 3:
        shape_candidates.append("8")
    return dict(status="VISUAL_EVIDENCE_ONLY", shape_candidates=shape_candidates,
                threshold_support=dict(hits), probability=None,
                sign_status="MINUS_COMPATIBLE" if (
                    hits["isolated_vertical_stroke"] >= 2
                    and hits["left_horizontal_stroke"] >= 3) else "UNKNOWN",
                details=per_threshold, executable=False)


def floor_value(text):
    normalized = vision.normalize_floor(str(text)) if text is not None else None
    return int(normalized) if normalized is not None else None


def function_word(variants,min_score):
    return any(len(t.strip())>=3 and t.strip().isalpha() and s>=min_score for t,s in variants)


def _clusters(indices, coordinates, tolerance):
    groups = []
    for i in sorted(indices, key=lambda k: coordinates[k]):
        if not groups or abs(coordinates[i]-np.mean([coordinates[j] for j in groups[-1]])) > tolerance:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def numeric_grid(buttons, anchors, excluded):
    """Estimate rows/columns from measured rims, with numeric OCR as region anchors."""
    indices = [i for i in range(len(buttons)) if i not in excluded]
    if len(indices) > 100 or not anchors:
        return None
    radius = float(np.median([buttons[i].ring.radius for i in anchors]))
    points = np.array([[b.ring.x, b.ring.y] for b in buttons], float)
    angles = []
    for i in anchors:
        for j in anchors:
            dx, dy = points[j]-points[i]
            if dx > 1.5*radius and abs(dy) < .45*dx and dx < 8*radius:
                angles.append(math.atan2(dy, dx))
    angle = float(np.median(angles)) if angles else 0.
    cs, sn = math.cos(angle), math.sin(angle)
    uv = points@np.array([[cs,-sn],[sn,cs]])
    anchor_cols = _clusters(list(anchors), uv[:,0], .85*radius)
    cols = [float(np.mean(uv[g,0])) for g in anchor_cols]
    # An entirely unread column can still be measured: require at least three
    # repeated rims aligned with at least two numeric-anchor rows, not one icon.
    for group in _clusters(indices,uv[:,0],.85*radius):
        cx=float(np.mean(uv[group,0]));distance=min(abs(cx-x) for x in cols)
        aligned=sum(any(abs(uv[i,1]-uv[j,1])<=.85*radius for j in anchors) for i in group)
        if len(group)>=3 and aligned>=2 and 1.8*radius<=distance<=5.8*radius:
            cols.append(cx)
    cols.sort()
    # A nearby icon is not a numeric slot merely because it has a circular rim.
    indices = [i for i in indices if min(abs(uv[i,0]-x) for x in cols) <= .7*radius
               and .65 <= buttons[i].ring.radius/radius <= 1.5]
    rows = _clusters(indices, uv[:,1], .85*radius)
    if not rows or not 1 <= len(cols) <= 6:
        return None
    row_y = [float(np.mean(uv[g,1])) for g in rows]
    if len(rows) >= 3:
        gaps = np.diff(row_y)
        if gaps.max() > 1.8*np.median(gaps) or gaps.min() < 1.3*radius:
            return None  # A separated function row or irregular panel needs review.
    cells = {}
    for r, group in enumerate(rows):
        for i in group:
            col = int(np.argmin([abs(uv[i,0]-x) for x in cols]))
            if (r,col) in cells:
                return None  # Never collapse two measured buttons into one slot.
            cells[r,col] = i
    orders = []
    for major in ('row','column'):
        for up in (False,True):
            for right in (False,True):
                key = (lambda cell: ((-cell[0] if up else cell[0]),(-cell[1] if right else cell[1])))
                if major == 'column':
                    key = lambda cell: ((-cell[1] if right else cell[1]),(-cell[0] if up else cell[0]))
                orders.append((f'{major}_y{"up" if up else "down"}_x{"left" if right else "right"}',
                               [cells[c] for c in sorted(cells,key=key)]))
    return dict(cells=cells,orders=orders,angle_deg=math.degrees(angle),
                row_count=len(rows),column_count=len(cols),indices=indices)


def layout_candidates(buttons, doors, glyphs, *, min_score=.70, floor_list=None, inventory_complete=False):
    """Keep every consistent scan model, including skipped floors. Never fabricate a rim.

    Consecutive labels are preferences. Conservative candidates also allow
    strictly increasing skipped floor values; glyph support is kept separately.
    """
    if floor_list is not None:
        if not floor_list or len(set(floor_list)) != len(floor_list) or any(abs(x)>99 for x in floor_list):
            raise ValueError('floor-list需要互不重复的-99..99整数')
        floor_list = sorted(floor_list)
    if inventory_complete and floor_list is None:
        raise ValueError('inventory-complete需要显式floor-list')
    excluded, anchors, conflicts, rows = set(), {}, set(), []
    for i,b in enumerate(buttons):
        value = floor_value(b.text)
        door = doors.get(i,{}).get('label','UNKNOWN')
        word = function_word(b.variants,min_score)
        reads = {floor_value(t) for t,s in b.variants if np.isfinite(s) and s>=min_score}-{None}
        if door != 'UNKNOWN' or word:
            excluded.add(i)
        elif len(reads)>1:
            conflicts.add(i)
        elif value is not None and b.confidence is not None and b.confidence>=min_score:
            anchors[i] = value
        rows.append(dict(index=i,button_candidate_center_px=list(b.ring.center),
                         radius_px=b.ring.radius,ocr_text=b.text,ocr_confidence=b.confidence,
                         visual_evidence=glyphs.get(i,{}),
                         status='DIRECT_OCR' if i in anchors else ('FUNCTION_OR_WORD' if i in excluded else
                            'OCR_NUMERIC_CONFLICT' if i in conflicts else 'UNKNOWN'),
                         floor_candidates=[str(value)] if i in anchors else [],preferred_candidates=[],
                         ocr_numeric_alternatives=[str(v) for v in sorted(reads)],
                         probability=None,executable=False))
    report = dict(status='INSUFFICIENT_ANCHORS',buttons=rows,models=[],anchors={str(k):v for k,v in anchors.items()},
                  assumptions=['Only measured current rims receive coordinates.',
                    'Monotonic scan order is a hypothesis, not a universal elevator rule.',
                    'Shape candidates include I/1 and B/8 ambiguities; no motion is authorized.'],
                  inventory_complete=inventory_complete)
    if len(anchors)<3:
        return report
    if len(set(anchors.values())) != len(anchors):
        report['status']='DUPLICATE_ANCHOR_LABELS';return report
    grid = numeric_grid(buttons,anchors,excluded)
    if grid is None or not set(anchors)<=set(grid['indices']):
        report['status']='IRREGULAR_OR_UNSUPPORTED_GRID';return report
    report['grid']={k:grid[k] for k in ('row_count','column_count','angle_deg','indices')}
    domains = [('provided_inventory',floor_list)] if floor_list is not None else [
        ('zero_allowed',list(range(-99,100))),('zero_absent',[v for v in range(-99,100) if v!=0])]
    possible = {i:set() for i in grid['indices']}
    preferred = {i:set() for i in grid['indices']}
    if inventory_complete and len(grid['indices'])!=len(floor_list):
        report['status']='INCOMPLETE_OR_MISMATCHED_GRID';return report
    outside=set(range(len(buttons)))-set(grid['indices'])-excluded
    for i in outside:rows[i]['status']='UNSUPPORTED_REGION'
    for order_name,order in grid['orders']:
        for domain_name,domain in domains:
            lookup = {v:j for j,v in enumerate(domain)}
            if any(v not in lookup for v in anchors.values()):continue
            fixed = sorted((p,lookup[anchors[i]]) for p,i in enumerate(order) if i in anchors)
            if not fixed:continue
            feasible = []
            for p,i in enumerate(order):
                # Domain ends are virtual anchors too: enough distinct floor
                # values must remain for all earlier/later measured buttons.
                lo = max([p]+[rank+p-at for at,rank in fixed if at<=p])
                hi = min([len(domain)-len(order)+p]+[rank-(at-p) for at,rank in fixed if at>=p])
                feasible.append((lo,hi))
            if any(lo>hi for lo,hi in feasible):continue
            if inventory_complete:
                if len(order)!=len(domain) or any(rank!=p for p,rank in fixed):continue
                feasible = [(p,p) for p in range(len(order))]
            starts = {rank-p for p,rank in fixed}
            consecutive = len(starts)==1 and 0<=min(starts) and min(starts)+len(order)<=len(domain)
            model = dict(order=order_name,domain=domain_name,allows_skipped_floors=not inventory_complete,
                         consecutive_also_fits=consecutive,button_order=order)
            report['models'].append(model)
            for p,(i,(lo,hi)) in enumerate(zip(order,feasible)):
                possible[i].update(domain[lo:hi+1])
                if consecutive:preferred[i].add(domain[min(starts)+p])
    report['status']='CANDIDATES_ONLY' if report['models'] else 'ANCHOR_LAYOUT_CONFLICT'
    for i in grid['indices']:
        if i in anchors:continue
        entry=rows[i]
        entry.update(layout_possible_floors=[str(v) for v in sorted(possible[i])],
                     preferred_candidates=[str(v) for v in sorted(preferred[i])],
                     floor_candidates=[str(v) for v in sorted(possible[i])],
                     surviving_models=len(report['models']))
        entry['requires_confirmation']=True
    return attach_glyph_candidates(report,glyphs)


def attach_glyph_candidates(report,glyphs):
    """Add pixel evidence without refitting the same grid/ordering hypotheses."""
    for entry in report['buttons']:
        i=entry['index'];entry['visual_evidence']=glyphs.get(i,{})
        if 'layout_possible_floors' not in entry:continue
        possible={int(v) for v in entry['layout_possible_floors']}
        shape={floor_value(v) for v in glyphs.get(i,{}).get('shape_candidates',[])}-{None}
        supported=possible&shape
        entry['shape_layout_agreement']=[str(v) for v in sorted(supported)]
        if len(entry['ocr_numeric_alternatives'])>1:
            entry['status']='OCR_NUMERIC_CONFLICT'
        elif shape and possible and not supported:
            entry['status']='SHAPE_LAYOUT_CONFLICT'
        elif supported:
            entry['status']='SHAPE_LAYOUT_CANDIDATE'
        elif possible:
            entry['status']='LAYOUT_REVIEW_REQUIRED'
        else:
            entry['status']='UNKNOWN'
        # A rank fit alone cannot tell whether a missing circle was undetected.
    return report


class PositionSupport:
    """Bounded floor-1 inference from observed glyphs AND numeric scan models.

    Raw OCR remains untouched. This does not assume that 0 is absent, fill a
    missing circle, or infer other labels solely from a consecutive sequence.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.votes = {}
        self.layout_cache = None

    @staticmethod
    def fit_models(buttons, doors, glyphs, anchors, index, min_score):
        buttons = [b if i in anchors else replace(b, text=None, confidence=None) for i,b in enumerate(buttons)]
        layout = layout_candidates(buttons, doors, glyphs, min_score=min_score)
        row = layout['buttons'][index]
        if row.get('shape_layout_agreement') != ['1'] or not layout['models']:
            return 0
        for model in layout['models']:
            order = model['button_order']
            if index not in order:return 0
            domain = [v for v in range(-99, 100) if model['domain'] != 'zero_absent' or v != 0]
            rank = {v:k for k,v in enumerate(domain)}
            p = order.index(index)
            fixed = [(k, rank[anchors[i]]) for k,i in enumerate(order) if i in anchors]
            lo = max([p]+[v+p-k for k,v in fixed if k <= p])
            hi = min([len(domain)-len(order)+p]+[v-(k-p) for k,v in fixed if k >= p])
            if not lo <= rank[1] <= hi:return 0
        return len(layout['models'])

    def update(self, result, min_score=.70, *, offline=False):
        metadata = result.tracking
        glyphs = metadata.get('glyph_evidence', {})
        evidence = metadata.get('region_evidence', {})
        doors = metadata.get('door_audit', {})
        def item(rows, index):
            return rows.get(index, rows.get(str(index), {}))
        candidates = []
        anchors = {}
        for i, button in enumerate(result.buttons):
            value = floor_value(button.text)
            if (value is not None and button.confidence is not None and button.confidence >= min_score
                    and button.observations >= 3 and item(doors, i).get('label', 'UNKNOWN') == 'UNKNOWN'):
                anchors[i] = value
            reads = {floor_value(t) for t,s in button.variants if math.isfinite(s) and s >= min_score}-{None}
            if (button.text is None and set(item(glyphs, i).get('shape_candidates', [])) == {'1'}
                    and reads <= {1} and not function_word(button.variants, min_score)
                    and item(doors, i).get('label', 'UNKNOWN') == 'UNKNOWN'
                    and item(evidence, i).get('track_id') is not None and button.ring.edge_support >= .85):
                candidates.append(i)
        if (result.errors or result.panel_bbox is None or len(candidates) != 1 or len(anchors) < 3
                or len(set(anchors.values())) != len(anchors) or 1 in anchors.values()
                or not offline and metadata.get('geometry_settled_frames', 0) < 3):
            self.reset(); return {}
        index = candidates[0]
        # Fit once for unchanged identities/anchors/strokes. Every update still
        # checks current ring geometry; appreciable drift or new pixels refit.
        key = (index, min_score, tuple((i, item(evidence, i).get('track_id'), anchors.get(i),
                    item(doors, i).get('label', 'UNKNOWN'), tuple(item(glyphs, i).get('shape_candidates', [])),
                    tuple(sorted({floor_value(t) for t,s in b.variants if math.isfinite(s) and s >= min_score}-{None})))
                    for i,b in enumerate(result.buttons)))
        cache = self.layout_cache
        same_geometry = (cache is not None and len(cache['rings']) == len(result.buttons)
                         and all(math.hypot(b.ring.x-r.x, b.ring.y-r.y) <= max(2., .1*r.radius)
                                 and .94 <= b.ring.radius/r.radius <= 1.06
                                 for b,r in zip(result.buttons, cache['rings'])))
        if cache is None or cache['key'] != key or not same_geometry:
            count = self.fit_models(result.buttons, doors, glyphs, anchors, index, min_score)
            cache = self.layout_cache = dict(key=key, rings=tuple(b.ring for b in result.buttons), models=count)
        # Every surviving order (including 0-present and skipped floors) must
        # permit 1 at this measured rim, rather than choosing a favorable order.
        if not cache['models']:
            self.votes = {}; return {}
        button = result.buttons[index]
        proof = item(evidence, index)
        track_id = proof['track_id']
        signature = tuple((i, item(evidence, i).get('track_id'), v) for i,v in sorted(anchors.items()))
        previous = self.votes.get(track_id)
        continued = (previous is not None and previous['signature'] == signature and proof.get('pixel_match')
                     and 0 < result.captured_at-previous['at'] <= .8
                     and math.hypot(button.ring.x-previous['ring'].x, button.ring.y-previous['ring'].y)
                         <= max(3., .15*previous['ring'].radius))
        count = previous['count']+1 if continued else 1
        self.votes = {track_id:dict(signature=signature, at=result.captured_at, count=count, ring=button.ring)}
        return {str(index):dict(label='1', source='GLYPH_LAYOUT', probability=None, ocr_confidence=None,
            status='STABLE_POSITION' if count >= 3 and not offline else 'POSITION_CANDIDATE',
            observations=count, track_id=track_id, pixel_match=bool(proof.get('pixel_match')),
            captured_at=result.captured_at, ring_center_px=list(button.ring.center),
            anchor_labels=[str(v) for v in sorted(anchors.values())], all_models_allow_label=True,
            surviving_models=cache['models'], coordinates_source='CURRENT_MEASURED_RIM')}


def region_patch(image,ring):
    """Canonical glyph pixels, excluding the rim and most braille."""
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    size=(max(8,round(1.32*ring.radius)),max(8,round(1.25*ring.radius)))
    patch=cv2.getRectSubPix(gray,size,(float(ring.x),float(ring.y-.16*ring.radius)))
    return cv2.GaussianBlur(cv2.resize(patch,(64,64)),(3,3),.5)


def patch_matches(reference,current):
    if reference is None or reference.shape!=current.shape or np.std(current)<5:return False
    ratio=float(np.std(current))/max(1.,float(np.std(reference)))
    if not .65<=ratio<=1.5:return False
    # Small registration compensates edge-refinement jitter. Correlation is
    # checked against the last OCR reference, not a drifting frame-to-frame mean.
    template=reference[3:-3,3:-3]
    return float(cv2.matchTemplate(current,template,cv2.TM_CCOEFF_NORMED).max())>=.97


def isolated_glyph_views(image,ring,shape_candidates=()):
    """Extract existing ink/strokes; never draw a minus or substitute I with 1."""
    r=ring.radius
    patch,box=vision.crop(image,(ring.x-.70*r,ring.y-.85*r,ring.x+.70*r,ring.y+.45*r))
    if patch is None:return [],None
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY);ox,oy=box[:2]
    views=[];bounds=[]
    for percentile in (87,92):
        threshold=max(np.percentile(gray,percentile),np.median(gray)+.3*(int(gray.max())-np.median(gray)))
        mask=np.uint8(gray>=threshold)*255
        _,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
        chosen=[k for k,(x,y,w,h,area) in enumerate(stats) if k and .45*r<h<1.2*r and area>.015*r*r
                and x>0 and y>0 and x+w<mask.shape[1] and y+h<mask.shape[0]
                and abs((centers[k,0]+ox-ring.x)/r)<.5]
        if not chosen:continue
        strokes=list(chosen)
        for k,(x,y,w,h,area) in enumerate(stats):
            if (k and k not in chosen and w>.2*r and h<.23*r and area>.008*r*r
                    and x>0 and y>0 and x+w<mask.shape[1] and y+h<mask.shape[0]
                    and any(abs(centers[k,1]-centers[t,1])<.3*r and
                            abs(centers[k,0]-centers[t,0])<.85*r for t in strokes)):
                chosen.append(k)  # Include only observed disconnected strokes.
        x1=min(stats[k,0] for k in chosen);y1=min(stats[k,1] for k in chosen)
        x2=max(stats[k,0]+stats[k,2] for k in chosen);y2=max(stats[k,1]+stats[k,3] for k in chosen)
        clean=np.uint8(~np.isin(labels,chosen))*255
        glyph=clean[y1:y2,x1:x2];height=64;width=max(3,round(glyph.shape[1]*height/glyph.shape[0]))
        glyph=cv2.resize(glyph,(width,height),interpolation=cv2.INTER_NEAREST)
        for margin in (.20,.45):
            view=cv2.copyMakeBorder(glyph,10,10,round(height*margin),round(height*margin),
                                    cv2.BORDER_CONSTANT,value=255)
            views.append(cv2.cvtColor(view,cv2.COLOR_GRAY2BGR))
        bounds.append((int(x1+ox),int(y1+oy),int(x2+ox),int(y2+oy)))
    if set(shape_candidates)=={'1'}:
        # Percentiles can retain a diagonal specular streak beside a thin 1.
        # Pick existing isolated vertical components by image topology, before
        # seeing OCR scores. Avoid completely filled rectangles and broken
        # fragments; no character is drawn or substituted.
        thin=[]
        thresholds=sorted(set(THRESHOLDS+(int(np.percentile(gray,87)),int(np.percentile(gray,92)))))
        for threshold in thresholds:
            _,labels,stats,centers=cv2.connectedComponentsWithStats(np.uint8(gray>=threshold)*255)
            for k,(x,y,w,h,area) in enumerate(stats):
                if not k:continue
                cx=(centers[k,0]+ox-ring.x)/r
                if not (-.12<=cx<=.30 and .60*r<=h<=1.10*r and .025*r<=w<=.18*r
                        and area>=.02*r*r and x>0 and y>0 and x+w<gray.shape[1] and y+h<gray.shape[0]):continue
                other_tall=any(j and j!=k and hh>.45*r and aa>.035*r*r and xx>0 and yy>0
                               and xx+ww<gray.shape[1] and yy+hh<gray.shape[0]
                               for j,(xx,yy,ww,hh,aa) in enumerate(stats))
                if other_tall:continue
                glyph=np.uint8(labels[y:y+h,x:x+w]!=k)*255
                glyph=cv2.resize(glyph,(max(3,round(w*64/h)),64),interpolation=cv2.INTER_NEAREST)
                tile=cv2.copyMakeBorder(glyph,10,10,22,22,cv2.BORDER_CONSTANT,value=255)
                quality=abs(area/(w*h)-.85)+.1*abs(w/r-.10)
                thin.append((quality,tile,(int(x+ox),int(y+oy),int(x+w+ox),int(y+h+oy))))
        thin.sort(key=lambda row:row[0]);picked=[];seen=set()
        for _,tile,rect in thin:
            key=tile.shape,tile.tobytes()
            if key not in seen:picked.append((cv2.cvtColor(tile,cv2.COLOR_GRAY2BGR),rect));seen.add(key)
            if len(picked)>=2:break
        if picked:
            views=views[:2]+[t for t,_ in picked]
            bounds=[rect for _,rect in picked]
    if not bounds:return [],None
    return views,(min(b[0] for b in bounds),min(b[1] for b in bounds),max(b[2] for b in bounds),max(b[3] for b in bounds))


class RegionOCR:
    """Bounded OCR batches and pixel-verified display memory, private to this experiment."""
    def __init__(self,original,budget=4,refresh_s=2.):
        self.original=original;self.budget=budget;self.refresh_s=refresh_s
        inner=original.original if isinstance(original,vision.GeneralizedOCR) else original
        self.retry_backend=inner if isinstance(inner,vision._OCRBatch) else None
        self.backend_info=original.backend_info;self.next_id=0;self.clear()

    def clear(self):
        self.memories=[];self.last_doors={};self.last_glyphs={};self.evidence={}
        self.origin=(0,0);self.now=0.;self.offline=False;self.ready=False
        self.last_inference_stats={};self.last_patch_candidates=0
        self.retry_audit={};self.retry_stats={}

    def retry_unread(self,image,rings,inferred,entries,selected):
        self.retry_audit={};self.retry_stats={}
        base=self.retry_backend
        if base is None:return inferred
        patches=[];owners=[];boxes={};shape_support={}
        anchor_x=[b.ring.x for b in inferred if b.text is not None]
        anchor_x.extend(m['ring'].x-self.origin[0] for m in entries
                        if m['best'] is not None and self.now-m['last_good']<=8.)
        for index,observation in enumerate(inferred):
            if observation.text is not None or function_word(observation.variants,base.min_score):continue
            reads={vision.normalize_floor(t) for t,s in observation.variants
                   if math.isfinite(s) and s>=base.min_score and vision.normalize_floor(t) is not None}
            if len(reads)>1:continue  # Preserve actual numeric conflicts.
            ring=rings[index];memory=entries[selected[index]]
            if memory['glyph'] is None:memory['glyph']=glyph_shape_evidence(image,ring.center,ring.radius)
            shapes=set(memory['glyph'].get('shape_candidates',[]));shape_support[index]=shapes
            if not reads and not shapes:continue
            # A wood-grain proposal can contain vertical strokes too. A
            # shape-only retry needs an observed numeric region, not a guessed
            # floor sequence. Existing OCR for other regions remains unchanged.
            if not reads and len(anchor_x)<3:continue
            if len(anchor_x)>=3 and not any(abs(ring.x-x)<.8*ring.radius for x in anchor_x):continue
            views,box=isolated_glyph_views(image,ring,shapes or reads)
            if not views:continue
            boxes[index]=box
            for patch in views:patches.append(patch);owners.append(index)
            if len(boxes)>=2:break  # At most two additional local retry regions.
        if not patches:return inferred
        frame=vision._FrameRecognition(base.engine,base.input_type)
        readings={index:[] for index in boxes}
        for index,reading in zip(owners,frame.recognize(patches)):readings[index].append(reading)
        self.retry_stats=dict(frame.stats,glyph_retry_buttons=len(boxes),glyph_retry_patches=len(patches))
        for index,views in readings.items():
            old=inferred[index]
            label,confidence=vision.choose_floor(list(old.variants)+views,base.min_score)
            memory=entries[selected[index]];prior_used=False
            current_numeric=[(vision.normalize_floor(t),float(s)) for t,s in list(old.variants)+views
                             if math.isfinite(s) and s>=base.min_score and vision.normalize_floor(t) is not None]
            # After two genuinely accepted reads of unchanged pixels, a fresh
            # strong reading may confirm the same label even if another crop
            # returns I/Y/empty. A conflicting number still cancels this path.
            if (label is None and memory.get('count',0)>=2 and memory.get('same_pixels',False)
                    and memory['best'] is not None):
                expected=memory['best'].text
                strong=[float(s) for t,s in views if vision.normalize_floor(t)==expected and math.isfinite(s) and s>=max(.80,base.min_score)]
                if strong and {t for t,_ in current_numeric}=={expected}:
                    label=expected;confidence=float(np.mean(strong));prior_used=True
            original_reads={vision.normalize_floor(t) for t,s in old.variants
                            if math.isfinite(s) and s>=base.min_score and vision.normalize_floor(t) is not None}
            retry_support=any(vision.normalize_floor(t)==label and math.isfinite(s) and s>=base.min_score for t,s in views)
            compatible=(label is not None and retry_support and
                        (label in original_reads if original_reads else label in shape_support[index]))
            self.retry_audit[str(selected[index])]=dict(original_variants=list(old.variants),retry_variants=views,
                accepted_text=label if compatible else None,used_prior_verified_ocr=bool(compatible and prior_used),
                source='ISOLATED_CURRENT_GLYPH',text_bbox_xyxy=boxes[index])
            if compatible:
                inferred[index]=replace(old,text=label,confidence=confidence,text_bbox=boxes[index],
                    variants=list(old.variants)+views,text_region_source='isolated_glyph_retry')
        return inferred

    def recognize(self,image,rings,box):
        old=self.memories;used=set();entries=[];patches=[];self.last_doors={};self.last_glyphs={};self.evidence={}
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.retry_audit={};self.retry_stats={}
        ox,oy=self.origin
        for i,ring in enumerate(rings):
            global_ring=replace(ring,x=ring.x+ox,y=ring.y+oy)
            matches=[(math.hypot(global_ring.x-m['ring'].x,global_ring.y-m['ring'].y),k) for k,m in enumerate(old)
                     if k not in used and .85<=global_ring.radius/m['ring'].radius<=1.18]
            distance,k=min(matches) if matches else (math.inf,None)
            memory=old[k] if k is not None and distance<.45*ring.radius else None
            if memory is not None:used.add(k)
            # Edge verification can vary radius by a few percent even in a
            # stationary view. Use the acquired radius for glyph sampling;
            # current rim geometry is still verified and returned separately.
            sample=replace(ring,radius=memory['sample_radius']) if memory is not None else ring
            patch=region_patch(gray,sample)
            same=memory is not None and patch_matches(memory['patch'],patch)
            if not same:
                self.next_id+=1
                patch=region_patch(gray,ring)
                door=classify_door(image,ring)
                memory=dict(id=self.next_id,ring=global_ring,patch=patch,door=door,
                            sample_radius=ring.radius,glyph=None,best=None,count=0,
                            last_attempt=-math.inf,last_good=-math.inf,attempts=0,raw=None)
            patches.append(patch)
            memory['ring']=global_ring;memory['same_pixels']=same
            memory['ocr_current']=False
            entries.append(memory)
        self.memories=entries
        eligible=[]
        for i,memory in enumerate(entries):
            if memory['door']['label']!='UNKNOWN':continue
            age=self.now-memory['last_attempt']
            due=(memory['attempts']==0 or (memory['count']<3 and age>=.12)
                 or (memory['count']>=3 and age>=self.refresh_s))
            # Unchanged unread glyphs do not improve by repeatedly submitting
            # identical crops. Changed pixels always create a fresh memory.
            if memory['best'] is None and memory['attempts']>=2:
                due=age>=min(8.,max(1.,2*self.refresh_s))
            if self.offline or (self.ready and due):eligible.append(i)
        eligible.sort(key=lambda i:(entries[i]['last_attempt'],entries[i]['id']))
        selected=eligible if self.offline else eligible[:self.budget]
        if selected:
            samples=[replace(rings[i],radius=entries[i]['sample_radius']) for i in selected]
            inferred=self.original.recognize(image,samples,box)
            if len(inferred)!=len(selected):raise RuntimeError('局部OCR返回数量与选区不一致')
            inferred=self.retry_unread(image,samples,list(inferred),entries,selected)
            for i,observation in zip(selected,inferred):
                memory=entries[i];memory['last_attempt']=self.now;memory['attempts']+=1;memory['raw']=observation
                memory['ocr_current']=observation.text is not None
                if observation.text is not None:
                    previous=memory['best']
                    memory['count']=memory['count']+1 if previous is not None and previous.text==observation.text else 1
                    memory['best']=observation;memory['best_ring']=rings[i];memory['last_good']=self.now
                    memory['patch']=patches[i]
                else:
                    threshold=getattr(self.original,'min_score',getattr(getattr(self.original,'original',None),'min_score',.7))
                    alternatives={vision.normalize_floor(t) for t,s in observation.variants
                                  if math.isfinite(s) and s>=threshold and vision.normalize_floor(t) is not None}
                    contradictory=(function_word(observation.variants,threshold) or
                                   bool(alternatives and (memory['best'] is None or alternatives!={memory['best'].text})))
                    if memory['count']<3 or contradictory:
                        memory['best']=None;memory['count']=0
                if observation.text is None and memory['glyph'] is None:
                    memory['glyph']=glyph_shape_evidence(image,rings[i].center,rings[i].radius)
        observations=[]
        for i,(ring,memory) in enumerate(zip(rings,entries)):
            door=memory['door'];self.last_doors[i]=dict(label=door['label'],score=door['score'],
                scores=door['scores'],center_px=list(ring.center),glyph_bbox_xyxy=None,
                source='PIXEL_MATCHED_CACHE' if memory['same_pixels'] else 'CURRENT_SHAPE')
            if memory['glyph'] is not None:self.last_glyphs[i]=memory['glyph']
            best=memory['best'];valid=best is not None and self.now-memory['last_good']<=8.
            if valid:
                original_ring=memory['best_ring'];bbox=best.text_bbox
                if bbox is not None:
                    ratio=ring.radius/original_ring.radius
                    bbox=tuple(round((value-offset)*ratio+new) for value,offset,new in
                               zip(bbox,(original_ring.x,original_ring.y)*2,(ring.x,ring.y)*2))
                stable=memory['count']>=3
                status=('STABLE' if stable else 'UNSTABLE') if memory['ocr_current'] else ('STABLE_CACHE' if stable else 'UNSTABLE_CACHE')
                observations.append(replace(best,ring=ring,text_bbox=bbox,status='SINGLE_SHOT' if self.offline else status,
                                            observations=memory['count']))
            else:
                raw=memory['raw']
                observations.append(vision.ButtonObservation(ring,variants=[] if raw is None else list(raw.variants),
                                    status='FUNCTION_SYMBOL' if door['label']!='UNKNOWN' else 'PENDING_OCR'))
            self.evidence[str(i)]=dict(track_id=memory['id'],source='CURRENT_OCR' if memory['ocr_current'] else
                                      'PIXEL_MATCHED_CACHE' if valid else 'NO_TEXT',
                                      ocr_observations=memory['count'],pixel_match=memory['same_pixels'],
                                      ocr_age_s=None if best is None else self.now-memory['last_good'])
        stats=dict(getattr(self.original,'last_inference_stats',{})) if selected else {}
        for name in ('batches','inferred_patches'):stats[name]=stats.get(name,0)+self.retry_stats.get(name,0)
        stats.update(glyph_retry_buttons=self.retry_stats.get('glyph_retry_buttons',0),
                     glyph_retry_patches=self.retry_stats.get('glyph_retry_patches',0))
        stats.update(ocr_buttons=len(selected),cached_buttons=sum(e['source']=='PIXEL_MATCHED_CACHE' for e in self.evidence.values()),
                     skipped_symbol_buttons=sum(m['door']['label']!='UNKNOWN' for m in entries),
                     geometry_settled=self.ready)
        self.last_inference_stats=stats;self.last_patch_candidates=len(selected)
        return observations


class EvidenceFrameResult(vision.FrameResult):
    def to_dict(self):
        record=super().to_dict()
        record['labels_source']=('CURRENT_OCR_WITH_SEPARATE_PIXEL_MATCHED_DISPLAY_CACHE'
                                if self.tracking.get('workflow_evidence') else
                                'CURRENT_OCR_OR_EXPLICIT_PIXEL_MATCHED_DISPLAY_CACHE')
        record['current_ocr_numbers']=[b.text for i,b in enumerate(self.buttons) if b.text is not None
                                       and self.tracking.get('region_evidence',{}).get(str(i),{}).get('source')=='CURRENT_OCR']
        return record


def panel_search_box(rings,shape):
    box=vision.group_box(rings,shape)
    if box is None:return None
    radius=float(np.median([r.radius for r in rings]))
    steps=grid_steps(rings)
    padx=max([.5*radius]+[abs(s[0])+.15*radius for s in steps])
    pady=max([3*radius]+[abs(s[1])+.15*radius for s in steps])
    # One extra row above/below the acquired rims. A missed outer row must
    # remain inside the search area, rather than being permanently cropped out.
    return vision.clip_box((box[0]-padx,box[1]-pady,box[2]+padx,box[3]+pady),shape)


def grid_steps(rings):
    steps=[]
    if len(rings)<4:return steps
    radius=float(np.median([r.radius for r in rings]))
    for axis in (0,1):
        vectors=[]
        for a in rings:
            candidates=[]
            for b in rings:
                dx,dy=b.x-a.x,b.y-a.y
                forward,side=(dx,dy) if axis==0 else (dy,dx)
                if 1.65*radius<forward<5*radius and abs(side)<.6*radius:
                    candidates.append((forward,(dx,dy)))
            if candidates:vectors.append(min(candidates)[1])
        if len(vectors)>=2:steps.append(np.median(vectors,axis=0))
    return steps


def confirm_rim(center,radius,distance):
    """A grid predicts only where to look; current curved edges confirm a rim."""
    h,w=distance.shape;x,y=center
    if x-1.3*radius<1 or y-1.3*radius<1 or x+1.3*radius>=w-1 or y+1.3*radius>=h-1:return None
    theta=np.linspace(0,2*np.pi,96,endpoint=False)
    offsets=np.array([(0,0),(-.12,0),(.12,0),(0,-.12),(0,.12)])*radius
    choices=np.array([(x+dx,y+dy,radius*s) for dx,dy in offsets for s in (.90,1.,1.10)])
    radial=choices[:,2,None,None]*np.linspace(.94,1.06,5)[None,:,None]
    xs=np.rint(choices[:,0,None,None]+radial*np.cos(theta)).astype(int)
    ys=np.rint(choices[:,1,None,None]+radial*np.sin(theta)).astype(int)
    supported=np.min(distance[ys,xs],axis=1)<max(1.7,.035*radius)
    scores=supported.mean(axis=1)
    quarters=supported.reshape(-1,4,24).mean(axis=2).min(axis=1)
    valid=(scores>=.78)&(quarters>=.50)
    if not valid.any():return None
    best=int(np.argmax(np.where(valid,scores,-1)))
    xx,yy,rr=choices[best]
    return vision.Ring(float(xx),float(yy),float(rr),float(scores[best]))


def recover_panel_rims(image,rings):
    """Bounded search near current rims; floor labels never enter this search."""
    info=dict(guided_rims_added=0,guided_probes=0)
    steps=grid_steps(rings)
    if not steps:return [],info
    radius=float(np.median([r.radius for r in rings]));points=[]
    xmin,xmax=min(r.x for r in rings),max(r.x for r in rings)
    ymin,ymax=min(r.y for r in rings),max(r.y for r in rings)
    for ring in rings:
        for step in steps:
            for sign in (-1,1):
                x,y=np.array([ring.x,ring.y])+sign*step
                if any(math.hypot(x-r.x,y-r.y)<.75*radius for r in rings):continue
                if any(math.hypot(x-p[0],y-p[1])<.75*radius for p in points):continue
                points.append((float(x),float(y)))
    points.sort(key=lambda p:0 if xmin<=p[0]<=xmax and ymin<=p[1]<=ymax else 1)
    box=panel_search_box(rings,image.shape);x0,y0,x1,y1=box
    work=image[y0:y1,x0:x1]
    distance=vision._edge_distance(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY));added=[]
    for point in points[:12]:
        rim=confirm_rim((point[0]-x0,point[1]-y0),radius,distance)
        if rim is not None:rim=replace(rim,x=rim.x+x0,y=rim.y+y0)
        if rim is not None and not any(math.hypot(rim.x-r.x,rim.y-r.y)<max(rim.radius,r.radius) for r in rings+added):
            added.append(rim)
        if len(added)>=4:break
    info.update(guided_probes=min(12,len(points)),guided_rims_added=len(added),
                guided_search_area_ratio=work.shape[0]*work.shape[1]/(image.shape[0]*image.shape[1]))
    return added,info


class FastPanelPerception(vision.PanelPerception):
    """Reuse existing detector/OCR once; track and verify only the acquired panel ROI."""
    def __init__(self,cfg,args):
        super().__init__(cfg,args)
        self.ocr=RegionOCR(self.ocr,getattr(args,'ocr_buttons_per_update',4),getattr(args,'ocr_refresh_s',2.))
        self.roi_bounds=None;self.settled=0;self.last_rings=[];self.lost=0;self.reset_requested=False
        self.refresh_every=120
        self.search_tick=0
        self.workflow_evidence=bool(getattr(args,'workflow_evidence',False))
        self.min_score=float(getattr(args,'ocr_min_score',.70))
        self.position_support=PositionSupport()

    def request_refresh(self):self.reset_requested=True

    def process(self,image,captured_at=None,offline=False):
        if self.reset_requested or offline:
            self.roi_bounds=None;self.settled=0;self.last_rings=[];self.layout.reset();self.ocr.clear();self.reset_requested=False
            self.search_tick=0
            self.position_support.reset()
        at=time.perf_counter() if captured_at is None else float(captured_at)
        box=None if offline else self.roi_bounds
        ox,oy=(0,0) if box is None else box[:2]
        work=image if box is None else image[box[1]:box[3],box[0]:box[2]]
        self.ocr.origin=(ox,oy);self.ocr.now=at;self.ocr.offline=offline;self.ocr.ready=offline or self.settled>=2
        model=self.model
        # Whole-frame YOLO is only an acquisition aid; current ROI rim evidence
        # is checked every update. A failed track triggers reacquisition.
        if box is not None:self.model=None
        elif not offline:self.calls=0
        try:result=super().process(work,captured_at=at,offline=offline)
        finally:self.model=model
        result.buttons=[replace(b,ring=replace(b.ring,x=b.ring.x+ox,y=b.ring.y+oy),
                         text_bbox=None if b.text_bbox is None else tuple(v+d for v,d in zip(b.text_bbox,(ox,oy,ox,oy)))) for b in result.buttons]
        if result.panel_bbox is not None:result.panel_bbox=tuple(v+d for v,d in zip(result.panel_bbox,(ox,oy,ox,oy)))
        result.image_shape=image.shape[:2]
        doors={}
        for i,door in self.ocr.last_doors.items():
            if i>=len(result.buttons):continue
            doors[i]=dict(door,center_px=list(result.buttons[i].ring.center))
        retry_audit={k:dict(row,text_bbox_xyxy=tuple(v+d for v,d in zip(row['text_bbox_xyxy'],(ox,oy,ox,oy))))
                     for k,row in self.ocr.retry_audit.items()}
        result.tracking.update(region_evidence=self.ocr.evidence if result.buttons else {},door_audit=doors,
            glyph_evidence=self.ocr.last_glyphs if result.buttons else {},analysis_roi_xyxy=box,
            glyph_retry_audit=retry_audit,
            processed_area_ratio=work.shape[0]*work.shape[1]/(image.shape[0]*image.shape[1]))
        current=[b.ring for b in result.buttons]
        result.tracking['guided_rims_added']=0
        if not offline:
            if len(current)>=4:
                self.search_tick+=1
                if self.search_tick%8==1:
                    start=time.perf_counter();added,info=recover_panel_rims(image,current)
                    result.tracking.update(info)
                    result.timings_ms['rim_recovery']=(time.perf_counter()-start)*1000
                    result.timings_ms['total']+=result.timings_ms['rim_recovery']
                    # New rim coordinates are measured now. OCR waits for the
                    # next frame, after this measured area enters the core ROI.
                    result.buttons.extend(vision.ButtonObservation(r,status='CURRENT_RIM_PENDING_OCR') for r in added)
                    current.extend(added)
                    if added:result.panel_bbox=vision.group_box(current,image.shape)
                same=(len(self.last_rings)>=4 and sum(any(math.hypot(r.x-p.x,r.y-p.y)<max(3.,.12*r.radius)
                      for p in self.last_rings) for r in current)>=.7*len(current))
                self.settled=self.settled+1 if same else 1;self.lost=0
                if self.roi_bounds is None or result.tracking.get('guided_rims_added',0):
                    self.roi_bounds=vision.group_box(current,image.shape)
                    x0,y0,x1,y1=self.roi_bounds
                    local=[replace(r,x=r.x-x0,y=r.y-y0) for r in current]
                    # Preserve the measured rim sizes when entering the ROI.
                    # Re-running Hough at the crop's different scale can shift
                    # a small digit's crop and undo previously good OCR.
                    self.layout.seed(image[y0:y1,x0:x1],local)
            else:
                self.settled=0;self.lost+=1;self.ocr.clear()
                if self.lost>=2:self.roi_bounds=None;self.layout.reset()
            self.last_rings=current
        result.tracking['geometry_settled_frames']=self.settled
        started=time.perf_counter()
        result.tracking['position_inference']=self.position_support.update(result,self.min_score,offline=offline)
        result.timings_ms['position_inference']=(time.perf_counter()-started)*1000
        result.timings_ms['total']+=result.timings_ms['position_inference']
        if self.workflow_evidence:
            # Display memory is not a new model reading. Keep current rings,
            # but let the existing workflow vote only on actual current OCR.
            cache={}
            for index,b in enumerate(result.buttons):
                evidence=result.tracking['region_evidence'].get(str(index),{})
                if evidence.get('source')=='PIXEL_MATCHED_CACHE':
                    cache[str(index)]=b.to_dict()
                    result.buttons[index]=replace(b,text=None,confidence=None,text_bbox=None,
                                                  variants=[],status='NO_TEXT',observations=0)
            result.tracking.update(workflow_evidence=True,display_label_cache=cache,ocr_min_score=self.min_score)
        return EvidenceFrameResult(**result.__dict__)


def analyze(image, result, *, floor_list=None, inventory_complete=False, min_score=.70):
    started=time.perf_counter()
    doors={};glyphs={}
    h,w=image.shape[:2];scale=min(1.,1600/max(h,w))
    work=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale)))) if scale<1 else image
    sx,sy=work.shape[1]/w,work.shape[0]/h
    # Bound feature work for full-resolution phone photos. All output pixels
    # are mapped back with actual independent x/y scales, like find_rings.
    measured=[replace(b.ring,x=b.ring.x*sx,y=b.ring.y*sy,
                      radius=b.ring.radius*math.sqrt(sx*sy)) for b in result.buttons]
    precomputed=result.tracking.get('door_audit')
    for i,b in enumerate(result.buttons):
        # An icon misread as a high-confidence digit must not become an anchor.
        # This is local geometry, without an additional OCR/model forward.
        if len(result.buttons)>100:break
        doors[i]=dict(precomputed.get(i,precomputed.get(str(i),{}))) if precomputed is not None else classify_door(work,measured[i])
        if not doors[i]:doors[i]=dict(label='UNKNOWN',score=0.,scores={'OPEN':0.,'CLOSE':0.},best=None)
        doors[i]['center_px']=list(b.ring.center)
        best=doors[i].get('best')
        if best is not None and scale<1:
            for triangle in best['triangles']:
                triangle['centroid']=(np.asarray(triangle['centroid'])/[sx,sy]).tolist()
                triangle['vertices']=(np.asarray(triangle['vertices'])/[sx,sy]).tolist()
            box=doors[i]['glyph_bbox_xyxy']
            doors[i]['glyph_bbox_xyxy']=[round(v/s) for v,s in zip(box,(sx,sy,sx,sy))]
    door_done=time.perf_counter()
    provisional=layout_candidates(result.buttons,doors,{},min_score=min_score,
                                  floor_list=floor_list,inventory_complete=inventory_complete)
    fitted_at=time.perf_counter()
    numeric_indices=set(provisional.get('grid',{}).get('indices',[]))
    precomputed_glyphs=result.tracking.get('glyph_evidence')
    for i,b in enumerate(result.buttons):
        if len(result.buttons)>100:break
        # Do not spend glyph passes on wood-grain circles, unsupported columns
        # or unlabelled function rows outside the measured numeric grid.
        if i in numeric_indices and b.text is None and doors.get(i,{}).get('label','UNKNOWN')=='UNKNOWN':
            if not function_word(b.variants,min_score):
                if precomputed_glyphs is not None:
                    glyphs[i]=precomputed_glyphs.get(i,precomputed_glyphs.get(str(i),{}))
                else:glyphs[i]=glyph_shape_evidence(work,measured[i].center,measured[i].radius)
    glyph_done=time.perf_counter()
    layout=attach_glyph_candidates(provisional,glyphs)
    for entry in layout['buttons']:
        i=entry['index']
        # Layout order alone must not turn telephone/alarm/unknown glyphs into
        # 9/10 or negative floor labels. Raw numerical evidence remains auditable.
        if (entry['ocr_text'] is None and not glyphs.get(i,{}).get('shape_candidates')
                and entry['status'] not in ('FUNCTION_OR_WORD','OCR_NUMERIC_CONFLICT')):
            entry.update(layout_audit_candidates=entry['floor_candidates'],
                         status='FUNCTION_OR_UNKNOWN',floor_candidates=[],preferred_candidates=[],
                         layout_possible_floors=[],shape_layout_agreement=[])
        evidence=result.tracking.get('region_evidence',{}).get(str(i),{})
        entry['text_evidence']=evidence
        entry['temporal_status']=result.buttons[i].status
        position=result.tracking.get('position_inference',{}).get(str(i),{})
        if position:
            entry['position_inference']=position
            entry['status']=position['status']
        if entry['status']=='DIRECT_OCR' and evidence.get('source')=='PIXEL_MATCHED_CACHE':
            entry['status']='STABLE_CACHE' if result.buttons[i].status=='STABLE_CACHE' else 'UNSTABLE_CACHE'
    elapsed=time.perf_counter()
    return dict(captured_at=result.captured_at,feature_image_scale_xy=[sx,sy],raw_ocr=result.to_dict(),door_symbols=[dict(index=i,**row,probability=None,executable=False)
                for i,row in doors.items() if row['label']!='UNKNOWN'],
                door_audit={str(i):row for i,row in doors.items()},layout=layout,
                timings_ms=dict(door=(door_done-started)*1000,glyph=(glyph_done-fitted_at)*1000,
                                layout=((fitted_at-door_done)+(elapsed-glyph_done))*1000,
                                extra_total=(elapsed-started)*1000),
                executable=False)


def target_candidates(record,floor):
    label=str(int(floor))
    candidates=[]
    for b in record['layout']['buttons']:
        if label not in b['floor_candidates']:continue
        support=('DIRECT_OCR' if b['status']=='DIRECT_OCR' else 'PIXEL_MATCHED_DISPLAY_CACHE'
                 if b['status'] in ('STABLE_CACHE','UNSTABLE_CACHE') else 'SHAPE_LAYOUT_SUPPORT'
                 if label in b.get('shape_layout_agreement',[]) else 'LAYOUT_POSSIBILITY_ONLY')
        candidates.append(dict(index=b['index'],center_px=b['button_candidate_center_px'],status=b['status'],
                               requested_floor_support=support,shape_supported=b.get('shape_layout_agreement',[]),
                               geometry_source='CURRENT_MEASURED_RIM',requires_confirmation=True,executable=False))
    order={'DIRECT_OCR':0,'PIXEL_MATCHED_DISPLAY_CACHE':1,'SHAPE_LAYOUT_SUPPORT':2,'LAYOUT_POSSIBILITY_ONLY':3}
    return sorted(candidates,key=lambda c:order[c['requested_floor_support']])


def draw(image,record,max_side=1400):
    h,w=image.shape[:2];scale=min(1.,max_side/max(h,w));display=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
    def put(text,xy,color,size=.52):
        pt=tuple(round(x*scale) for x in xy)
        cv2.putText(display,text,pt,cv2.FONT_HERSHEY_SIMPLEX,size,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(display,text,pt,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    door_indices={d['index'] for d in record['door_symbols']}
    for b in record['layout']['buttons']:
        x,y=b['button_candidate_center_px'];r=b['radius_px'];label=b['ocr_text'];color=(80,235,80)
        if b['index'] in door_indices:
            cv2.circle(display,(round(x*scale),round(y*scale)),max(1,round(r*scale)),(255,240,0),2,cv2.LINE_AA)
            continue  # OPEN/CLOSE below; do not simultaneously label UNREAD.
        if b['status'] in ('FUNCTION_OR_WORD','FUNCTION_OR_UNKNOWN'):
            label=None;color=(160,160,160)
        if label is None:
            choices=b.get('shape_layout_agreement',[]);color=(0,195,255)
            label='?='+','.join(choices) if len(choices)<=3 and choices else 'UNREAD'
            if b['status']=='OCR_NUMERIC_CONFLICT':label='? OCR CONFLICT '+','.join(b['ocr_numeric_alternatives'])
            elif b['status']=='SHAPE_LAYOUT_CONFLICT':label='? SHAPE/LAYOUT CONFLICT'
            position=b.get('position_inference',{})
            if position:
                label=position['label']+(' P' if position['status']=='STABLE_POSITION' else ' ?P')
                color=(255,160,60)  # Separate from green OCR and cyan door symbols.
        else:
            if b.get('temporal_status') in ('UNSTABLE','UNSTABLE_CACHE'):label=str(label)+' ~'
            if b.get('text_evidence',{}).get('source')=='PIXEL_MATCHED_CACHE':label=str(label)+' C'
        cv2.circle(display,(round(x*scale),round(y*scale)),max(1,round(r*scale)),color,2,cv2.LINE_AA)
        put(str(label),(max(0,x-r),max(22,y-r-8)),color)
    for d in record['door_symbols']:
        x,y=d['center_px'];put(d['label']+f" {d['score']:.2f}",(x-35,y-15),(255,240,0))
        cv2.drawMarker(display,(round(x*scale),round(y*scale)),(255,240,0),cv2.MARKER_CROSS,14,2)
    extra=record['timings_ms']['extra_total']
    tracking=record['raw_ocr'].get('tracking',{});stats=tracking.get('ocr_inference',{})
    total=record['raw_ocr']['timings_ms'].get('total',0)+extra
    for j,line in enumerate([record.get('legend','EXPERIMENT: OCR / C=pixel-verified cache / ?=shape support / blue=door'),
                             f"process={total:.1f}ms extra={extra:.1f}ms OCR regions={stats.get('ocr_buttons','all')} cached={stats.get('cached_buttons',0)} ROI={tracking.get('processed_area_ratio',1):.0%}; "+
                             record.get('mode_label',('ARM HOLD / NO PRESS' if record.get('head_up_enabled') else 'NO ROBOT'))]):
        put(line,(10,(24+24*j)/scale),(255,255,255),.48)
    return display


def draw_workflow_result(image,result,min_score=None):
    """Display-only reconstruction; never writes cached labels to motion data."""
    buttons=list(result.buttons)
    if min_score is None:min_score=float(result.tracking.get('ocr_min_score',.70))
    for key,cached in result.tracking.get('display_label_cache',{}).items():
        index=int(key)
        if not 0<=index<len(buttons):continue
        evidence=result.tracking.get('region_evidence',{}).get(key,{})
        if evidence.get('source')!='PIXEL_MATCHED_CACHE' or not evidence.get('pixel_match'):continue
        b=buttons[index]
        buttons[index]=replace(b,text=cached['text'],confidence=cached['confidence'],
            text_bbox=cached['text_bbox_xyxy'],variants=[(v['text'],v['confidence']) for v in cached['ocr_variants']],
            status=cached['status'],observations=cached['consistent_observations'])
    visible=replace(result,buttons=buttons,tracking=dict(result.tracking,workflow_evidence=False))
    record=analyze(image,visible,min_score=min_score)
    record.update(legend='YOLO/OCR | C=pixel cache | P=glyph+layout (not OCR) | ?=candidate | blue=door',mode_label='MAIN ELEVATOR')
    return draw(image,record,max_side=max(image.shape[:2]))


