"""Independent two-key call-panel vision. Upper=UP/lower=DOWN by layout.

No OCR/model initialization, camera, robot, or changes to numeric-panel rules.
Coordinates are measured rim candidates, not calibrated pressing targets.
"""
from __future__ import annotations

from dataclasses import dataclass,field
import math
import time
import cv2
import numpy as np
from . import perception as vision
from .panel_261112 import confirm_rim


def _valid_rims(rings,shape):
    h,w=shape[:2];kept=[]
    for r in sorted(rings,key=lambda r:-r.edge_support):
        if not np.isfinite([r.x,r.y,r.radius,r.edge_support]).all() or r.radius<6 or r.edge_support<.80:continue
        if r.x-r.radius<1 or r.y-r.radius<1 or r.x+r.radius>=w-1 or r.y+r.radius>=h-1:continue
        if any(math.hypot(r.x-p.x,r.y-p.y)<.65*max(r.radius,p.radius) for p in kept):continue
        kept.append(r)
    return kept


def select_pair(rings,shape,roi=None):
    """Select a UNIQUE isolated vertical pair, without naming arrow glyphs."""
    measured=_valid_rims(rings,shape)
    if len(measured)>80:return None,'CLUTTERED',{}
    eligible=[r for r in measured if roi is None or roi[0]<=r.x<roi[2] and roi[1]<=r.y<roi[3]]
    choices=[];extra=False
    for i,first in enumerate(eligible):
        for second in eligible[i+1:]:
            top,bottom=sorted((first,second),key=lambda r:r.y)
            radius=(top.radius+bottom.radius)/2
            dx,dy=bottom.x-top.x,bottom.y-top.y
            gap=math.hypot(dx,dy)
            ratio=min(top.radius,bottom.radius)/max(top.radius,bottom.radius)
            angle=abs(math.degrees(math.atan2(dx,dy)))
            if ratio<.75 or not 2.4<=gap/radius<=7. or angle>18.:continue
            axis=np.array([dx,dy])/gap
            isolated=True
            for other in measured:
                if other is top or other is bottom:continue
                if not .65<=other.radius/radius<=1.5:continue
                delta=np.array([other.x-top.x,other.y-top.y])
                along=float(delta@axis);across=abs(float(delta[0]*axis[1]-delta[1]*axis[0]))
                if -gap-.5*radius<=along<=2*gap+.5*radius and across<=4*radius:
                    isolated=False;extra=True;break
            if isolated:
                choices.append(((top,bottom),dict(radius_ratio=ratio,separation_radii=gap/radius,
                    tilt_deg=angle,min_edge_support=min(top.edge_support,bottom.edge_support))))
    if len(choices)==1:return choices[0][0],'PAIR_FOUND',choices[0][1]
    return None,('AMBIGUOUS' if len(choices)>1 else 'EXTRA_BUTTONS' if extra else 'NO_PAIR'),{}


def _pair_box(pair,shape,padding=2.):
    return vision.clip_box((min(r.x-padding*r.radius for r in pair),min(r.y-padding*r.radius for r in pair),
                            max(r.x+padding*r.radius for r in pair),max(r.y+padding*r.radius for r in pair)),shape)


def _gray(image,box):
    x0,y0,x1,y1=box;patch=image[y0:y1,x0:x1]
    scale=min(1.,480/max(patch.shape[:2]))
    ww,hh=max(1,round(patch.shape[1]*scale)),max(1,round(patch.shape[0]*scale))
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    if gray.shape!=(hh,ww):gray=cv2.resize(gray,(ww,hh),interpolation=cv2.INTER_AREA)
    return gray,ww/patch.shape[1],hh/patch.shape[0]


def _context_box(pair,shape):
    top,bottom=pair;r=(top.radius+bottom.radius)/2;gap=bottom.y-top.y
    return vision.clip_box((min(top.x,bottom.x)-4*r,top.y-gap-r,
                            max(top.x,bottom.x)+4*r,bottom.y+gap+r),shape)


def _context(image,box):
    x0,y0,x1,y1=box
    gray=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    thumb=cv2.resize(gray,(96,96),interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb=cv2.GaussianBlur(thumb,(3,3),0)
    return (thumb-float(thumb.mean()))/max(8.,float(thumb.std()))


def _context_changed(current,reference):
    # A small new button can disappear in a whole-region average. Check
    # 16x16 tiles too; normalized luminance suppresses simple exposure gain.
    delta=np.abs(current-reference)
    return bool(delta.mean()>.18 or delta.reshape(6,16,6,16).mean(axis=(1,3)).max()>.65)


def _large_rims(image,max_side):
    """Bounded fallback for close-up rims beyond the shared finder radius cap."""
    h,w=image.shape[:2];scale=min(1.,max_side/max(h,w))
    ww,hh=max(1,round(w*scale)),max(1,round(h*scale));sx,sy=ww/w,hh/h
    gray=cv2.cvtColor(cv2.resize(image,(ww,hh),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
    enhanced=cv2.createCLAHE(2.,(8,8)).apply(gray)
    blur=cv2.GaussianBlur(enhanced,(5,5),1);short=min(ww,hh)
    raw=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT_ALT,dp=1.5,minDist=max(10,short*.04),
        param1=200,param2=.85,minRadius=max(6,round(short*.10)),maxRadius=max(8,round(short*.22)))
    if raw is None:return []
    distance=cv2.distanceTransform(255-cv2.Canny(blur,70,160),cv2.DIST_L2,3)
    out=[]
    for x,y,radius in raw[0]:
        verified=vision.verify_ring(vision.Ring(float(x),float(y),float(radius),0.),distance,minimum=.85)
        if verified is not None:
            out.append(vision.Ring(verified.x/sx,verified.y/sy,verified.radius/math.sqrt(sx*sy),verified.edge_support))
    return out


@dataclass
class CallPanelResult:
    captured_at: float|None
    image_shape: tuple
    state: str='NO_PAIR'
    reason: str='NO_PAIR'
    geometry_source: str='NONE'
    pair: tuple=()
    metrics: dict=field(default_factory=dict)
    stable_observations: int=0
    tracking_roi: tuple|None=None
    hough_this_frame: bool=False
    hough_calls: int=0
    acquisition_side: int|None=None
    acquisition_methods: tuple=()
    timings_ms: dict=field(default_factory=dict)
    generation: int=0

    def to_dict(self):
        buttons=[dict(direction=label,rim_candidate_px=list(r.center),radius_px=r.radius,
                      edge_support=r.edge_support,bbox_xyxy=list(vision.clip_box((r.x-r.radius,r.y-r.radius,
                          r.x+r.radius,r.y+r.radius),self.image_shape)),direction_source='IMAGE_VERTICAL_ORDER')
                 for label,r in zip(('UP','DOWN'),self.pair)]
        return dict(branch='TWO_KEY_CALL_PANEL',captured_at=self.captured_at,image_shape_hw=list(self.image_shape),
            state=self.state,reason=self.reason,geometry_source=self.geometry_source,
            buttons=buttons,pair_metrics=self.metrics,stable_observations=self.stable_observations,
            tracking_roi_xyxy=self.tracking_roi,hough_this_frame=self.hough_this_frame,hough_calls=self.hough_calls,
            full_search_max_side=self.acquisition_side,
            acquisition_methods=list(self.acquisition_methods),
            timings_ms=self.timings_ms,generation=self.generation,valid_current=bool(self.pair),
            direction_policy='Upper=UP, lower=DOWN; upright camera and standard two-key layout assumed.',
            actions_enabled=False)


class CallPanelDetector:
    def __init__(self,*,max_side=1200,required=3,refresh_s=2.,acquire_interval_s=.35,max_gap_s=.8,roi=None):
        if not 320<=max_side<=1600 or not 3<=required<=30 or int(required)!=required:
            raise ValueError('max_side需320..1600，required需3..30整数')
        if not all(math.isfinite(v) and v>0 for v in (refresh_s,acquire_interval_s,max_gap_s)):
            raise ValueError('刷新/获取间隔与最大帧间隔必须为有限正数')
        if roi is not None and (len(roi)!=4 or not np.isfinite(roi).all() or roi[2]<=roi[0] or roi[3]<=roi[1]):
            raise ValueError('ROI需为有效的x1 y1 x2 y2')
        self.max_side=int(max_side);self.required=int(required);self.refresh_s=refresh_s
        self.acquire_interval_s=acquire_interval_s;self.max_gap_s=max_gap_s;self.user_roi=roi
        self.hough_calls=0;self.reset()

    def reset(self):
        self.pair=None;self.roi=None;self.previous=None;self.shape=None;self.reference_radii=None
        self.context_box=None;self.context=None;self.anchor=None;self.streak=0
        self.last_at=-math.inf;self.last_valid=-math.inf;self.next_acquire=-math.inf;self.last_audit=-math.inf
        self.generation=None
        self.search_sizes=sorted(set((min(640,self.max_side),min(960,self.max_side),self.max_side)))
        self.search_index=0;self.audit_side=self.search_sizes[0]

    def _seed(self,image,pair,reference_radii=None):
        box=self.roi
        if box is None or any(r.x-box[0]<1.4*r.radius or box[2]-r.x<1.4*r.radius or
                              r.y-box[1]<1.4*r.radius or box[3]-r.y<1.4*r.radius for r in pair):
            box=_pair_box(pair,image.shape)
        self.roi=box;self.previous=_gray(image,box)[0];self.pair=pair
        self.reference_radii=reference_radii or tuple(r.radius for r in pair)

    def _track(self,image):
        gray,sx,sy=_gray(image,self.roi);x0,y0,_,_=self.roi;rs=math.sqrt(sx*sy)
        local=[vision.Ring((r.x-x0)*sx,(r.y-y0)*sy,radius*rs,r.edge_support)
               for r,radius in zip(self.pair,self.reference_radii)]
        mask=np.zeros(gray.shape,np.uint8)
        for r in local:
            cv2.circle(mask,r.center,max(2,round(r.radius*1.18)),255,-1)
            cv2.circle(mask,r.center,max(1,round(r.radius*.65)),0,-1)
        transform=None
        points=cv2.goodFeaturesToTrack(self.previous,maxCorners=80,qualityLevel=.015,minDistance=3,mask=mask,blockSize=5)
        if points is not None and len(points)>=6:
            following,status,error=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None,winSize=(21,21),maxLevel=2)
            if following is not None and status is not None:
                backward,ok,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,following,None,winSize=(21,21),maxLevel=2)
                if backward is not None and ok is not None:
                    valid=(status[:,0]>0)&(ok[:,0]>0)&(np.linalg.norm(backward[:,0]-points[:,0],axis=1)<1.2)
                    if error is not None:valid&=error[:,0]<30
                    if valid.sum()>=6:
                        candidate,inliers=cv2.estimateAffinePartial2D(points[valid,0],following[valid,0],
                            method=cv2.RANSAC,ransacReprojThreshold=2.)
                        if candidate is not None and inliers is not None and inliers.mean()>=.70:
                            scale=math.hypot(candidate[0,0],candidate[1,0])
                            angle=abs(math.degrees(math.atan2(candidate[1,0],candidate[0,0])))
                            if .9<=scale<=1.1 and angle<=15:transform=candidate
        distance=vision._edge_distance(gray)
        for matrix,source in ((transform,'LOCAL_FLOW_EDGES'),(None,'LOCAL_EDGE_RECHECK')):
            if matrix is None and source=='LOCAL_FLOW_EDGES':continue
            measured=[];reference=[]
            for r in local:
                if matrix is not None:
                    xy=matrix@np.array([r.x,r.y,1.]);scale=math.hypot(matrix[0,0],matrix[1,0])
                    predicted=vision.Ring(float(xy[0]),float(xy[1]),r.radius*scale,0.)
                else:predicted=r
                current=vision.verify_ring(predicted,distance,minimum=.80)
                if current is None:current=confirm_rim((predicted.x,predicted.y),predicted.radius,distance)
                if current is None:break
                measured.append(vision.Ring(current.x/sx+x0,current.y/sy+y0,current.radius/rs,current.edge_support))
                reference.append(predicted.radius/rs)
            if len(measured)==2:
                pair,_,metrics=select_pair(measured,image.shape,self.user_roi)
                if pair is not None:return pair,source,metrics,tuple(reference)
        return None,'NONE',{},None

    def process(self,image,captured_at=None,*,offline=False,generation=0):
        tick=time.perf_counter()
        try:at=tick if captured_at is None else float(captured_at)
        except (TypeError,ValueError):at=math.nan
        shape=image.shape[:2] if isinstance(image,np.ndarray) and image.ndim>=2 else (0,0)
        result=CallPanelResult(at if math.isfinite(at) else None,shape,generation=generation)
        if not math.isfinite(at):
            self.reset();result.state=result.reason='INVALID_TIMESTAMP';return result
        if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or min(shape)<24:
            self.reset();result.state=result.reason='INVALID_FRAME';return result
        if offline or self.generation!=generation or self.shape!=shape:
            self.reset();self.generation=generation;self.shape=shape
        if at<=self.last_at:
            self.streak=0;self.anchor=None;result.state=result.reason='STALE_FRAME';return result
        self.last_at=at
        if at-self.last_valid>self.max_gap_s:
            # Old locations are only search hints. A resumed frame must prove
            # both rims again and accumulate a new stable streak.
            self.streak=0;self.anchor=None
        pair=None;metrics={};source='NONE';reason='WAITING_FOR_ACQUISITION';reference=None
        start=time.perf_counter()
        if self.pair is not None:pair,source,metrics,reference=self._track(image)
        result.timings_ms['tracking']=(time.perf_counter()-start)*1000
        context_changed=bool(pair is not None and self.context is not None and
                             _context_changed(_context(image,self.context_box),self.context))
        audit=pair is not None and (at-self.last_audit>=self.refresh_s or context_changed)
        if audit or pair is None and at>=self.next_acquire:
            side=self.audit_side if audit else self.search_sizes[self.search_index]
            start=time.perf_counter();raw=vision.find_rings(image,side);self.hough_calls+=1
            result.acquisition_methods=('GRADIENT',)
            selected,reason,found=select_pair(raw+list(pair or ()),image.shape,self.user_roi)
            if selected is None and reason=='NO_PAIR':
                # Do not replace a multi-button/ambiguous rejection with a
                # more permissive search. Only close-up misses get this pass.
                raw+=_large_rims(image,side);self.hough_calls+=1
                result.acquisition_methods+=('LARGE_GRADIENT_ALT',)
                selected,reason,found=select_pair(raw,image.shape,self.user_roi)
            finished=at+(time.perf_counter()-start)
            self.next_acquire=finished+self.acquire_interval_s;self.last_audit=finished
            result.acquisition_side=side
            result.hough_this_frame=True;result.timings_ms['acquisition']=(time.perf_counter()-start)*1000
            if selected is None:
                pair=None
                if reason in ('EXTRA_BUTTONS','AMBIGUOUS','CLUTTERED'):
                    self.pair=self.roi=self.previous=None;self.search_index=0
                else:self.search_index=(self.search_index+1)%len(self.search_sizes)
            else:
                if pair is None:
                    pair=selected;metrics=found;source='GLOBAL_HOUGH';self.audit_side=side
                self.context_box=_context_box(pair,image.shape);self.context=_context(image,self.context_box)
        if pair is None:
            self.streak=0;self.anchor=None;result.reason=reason
            result.state=reason if reason in ('EXTRA_BUTTONS','AMBIGUOUS','CLUTTERED') else 'NO_PAIR'
        else:
            if context_changed:self.anchor=None;self.streak=0
            close=(self.anchor is not None and all(math.hypot(r.x-a.x,r.y-a.y)<=max(3.,.15*a.radius)
                   and .9<=r.radius/a.radius<=1.1 for r,a in zip(pair,self.anchor)))
            if close:self.streak+=1
            else:self.anchor=pair;self.streak=1
            self.last_valid=at;self._seed(image,pair,reference)
            result.pair=pair;result.metrics=metrics;result.geometry_source=source
            result.state='SINGLE_SHOT' if offline else 'STABLE' if self.streak>=self.required else 'UNSTABLE'
            result.reason='UPPER_IS_UP_LOWER_IS_DOWN';result.stable_observations=self.streak
            result.tracking_roi=self.roi
        result.hough_calls=self.hough_calls;result.timings_ms['total']=(time.perf_counter()-tick)*1000
        return result


def draw_result(image,result,max_side=1280):
    h,w=image.shape[:2];scale=min(1.,max_side/max(h,w))
    display=cv2.resize(image,(round(w*scale),round(h*scale)));sx,sy=display.shape[1]/w,display.shape[0]/h
    def put(text,xy,color=(255,255,255),size=.55):
        pt=(round(xy[0]*sx),round(xy[1]*sy))
        cv2.putText(display,text,pt,cv2.FONT_HERSHEY_SIMPLEX,size,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(display,text,pt,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    for direction,r in zip(('UP','DOWN'),result.pair):
        color=(50,230,50) if result.state=='STABLE' else (0,200,255)
        center=(round(r.x*sx),round(r.y*sy));cv2.circle(display,center,round(r.radius*math.sqrt(sx*sy)),color,2,cv2.LINE_AA)
        cv2.drawMarker(display,center,(255,100,255),cv2.MARKER_CROSS,12,2)
        put(f'{direction} ({r.center[0]}, {r.center[1]})',(max(5,r.x-r.radius),max(65/sy,r.y-r.radius-10)),color)
    put('CALL PANEL | upper=UP / lower=DOWN | upright layout assumed',(8/sx,24/sy))
    put(f'{result.state} {result.stable_observations} | {result.geometry_source} | process={result.timings_ms.get("total",0):.1f}ms',
        (8/sx,48/sy))
    put('NO ROBOT | Q/Esc=quit R=reset',(8/sx,(display.shape[0]-12)/sy))
    return display
