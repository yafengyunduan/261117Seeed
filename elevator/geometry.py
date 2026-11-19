"""Button-center RGB-D geometry using the same camera/base transform as main.py."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import math
import unicodedata
import re
import cv2
import numpy as np
from utils.camera_utils import compose_cam_to_base_transform
from utils.ordinary_grasp import get_depth_mm
from utils.transforms import pose6d_to_mat4

if TYPE_CHECKING:
    from .motion import ArmFeedback


@dataclass(frozen=True)
class Snapshot:
    frame_id: int
    captured_at: float
    color: np.ndarray
    depth_mm: np.ndarray
    K: np.ndarray
    D: np.ndarray
    T_base_end: np.ndarray
    generation: int = 0
    arm_feedback: ArmFeedback | None = None


@dataclass(frozen=True)
class ButtonPoint:
    snapshot: Snapshot
    button: object
    point_camera: np.ndarray
    point_base: np.ndarray


@dataclass(frozen=True)
class ButtonSurface(ButtonPoint):
    outward_base: np.ndarray
    measured_outward_base: np.ndarray
    plane_rms_m: float
    plane_inliers: int


def normalize_floor_id(value):
    text=unicodedata.normalize("NFKC",str(value)).strip().replace("−","-")
    if not re.fullmatch(r"-?[0-9]{1,2}",text):
        raise ValueError("请输入整数楼层")
    return str(int(text))


def position_floor_indices(result, floor):
    """Explicit non-OCR support; positions always belong to current measured rims."""
    floor=normalize_floor_id(floor)
    if floor != '1':return set()
    metadata=getattr(result,'tracking',{})
    out=set()
    for key,proof in metadata.get('position_inference',{}).items():
        try:index=int(key)
        except (TypeError,ValueError):continue
        if not 0<=index<len(result.buttons):continue
        button=result.buttons[index]
        evidence=metadata.get('region_evidence',{}).get(str(index),{})
        glyphs=metadata.get('glyph_evidence',{})
        glyph=glyphs.get(index,glyphs.get(str(index),{}))
        if (proof.get('label')!=floor or proof.get('source')!='GLYPH_LAYOUT'
                or proof.get('status')!='STABLE_POSITION' or proof.get('observations',0)<3
                or not proof.get('pixel_match') or not evidence.get('pixel_match')
                or proof.get('track_id') is None or proof['track_id']!=evidence.get('track_id')
                or proof.get('captured_at')!=result.captured_at
                or proof.get('ring_center_px')!=list(button.ring.center)
                or not proof.get('all_models_allow_label') or proof.get('surviving_models',0)<1
                or len(set(proof.get('anchor_labels',[])))<3
                or set(glyph.get('shape_candidates',[]))!={'1'} or button.ring.edge_support<.85):continue
        if button.text is not None and normalize_floor_id(button.text)!=floor:continue
        out.add(index)
    return out


def select_floor(result, tracker, floor, min_confidence=.8, now=None):
    """Keep current OCR or proven label memory tied to one CURRENT rim."""
    floor=normalize_floor_id(floor)
    def matches(b):
        try:return normalize_floor_id(b.text)==floor and b.confidence is not None and b.confidence>=min_confidence
        except ValueError:return False
    raw=[i for i,b in enumerate(result.buttons) if b.text is not None and matches(b)]
    # Display memory cannot authorize motion, but a second currently verified
    # rim with that label remains an ambiguity even while its OCR is skipped.
    inferred=position_floor_indices(result,floor)
    visible=set(raw)|inferred
    metadata=getattr(result,'tracking',{})
    if metadata.get('workflow_evidence'):
        for key,cached in metadata.get('display_label_cache',{}).items():
            evidence=metadata.get('region_evidence',{}).get(key,{})
            try:
                valid=(evidence.get('pixel_match') and evidence.get('source')=='PIXEL_MATCHED_CACHE'
                       and normalize_floor_id(cached['text'])==floor
                       and math.isfinite(float(cached['confidence'])) and cached['confidence']>=min_confidence)
                index=int(key)
            except (KeyError,ValueError,TypeError):continue
            if valid and 0<=index<len(result.buttons):visible.add(index)
    if len(visible)>1:
        raise ValueError(f"楼层{floor}对应多个圆环，不能确定目标")
    matches={i for i in raw if result.buttons[i].status=="STABLE"}
    if inferred and (tracker is None or getattr(getattr(tracker,'panel',None),'stable',False)):
        matches.update(inferred)
    if tracker is not None:
        matches.update(a.observation_index for a in tracker.remembered_associations(result,now=now) if a.label==floor)
    if raw and matches and raw[0] not in matches:
        raise ValueError(f"楼层{floor}的当前文字与缓存圆环冲突")
    if len(matches)!=1:
        raise ValueError(f"楼层{floor}尚无唯一稳定的当前圆环；请等待或重新输入")
    return result.buttons[next(iter(matches))]


def point_from_button(snapshot, button, hand_eye=None, cfg=None, *, min_depth_samples=1):
    """Median depth at the rim center; NEVER use the OCR text center."""
    u,v=button.ring.center
    depth=np.asarray(snapshot.depth_mm)
    if depth.ndim!=2 or not (0<=u<depth.shape[1] and 0<=v<depth.shape[0]):
        raise ValueError("圆环中心不在当前深度图中")
    K=np.asarray(snapshot.K,float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError("相机内参无效")
    if min_depth_samples > 1:
        samples=depth[max(0,v-3):v+4,max(0,u-3):u+4]
        valid=samples[np.isfinite(samples) & (samples>=50) & (samples<=1000)]
        if valid.size < min_depth_samples:
            raise ValueError("圆环中心有效深度不足")
        z=float(np.median(valid))/1000.
    else:
        z=get_depth_mm(depth,u,v,roi_size=7)/1000.
    if not math.isfinite(z) or not .05<=z<=1.:
        raise ValueError("圆环中心没有有效深度")
    D=np.asarray(snapshot.D,float)
    xy=cv2.undistortPoints(np.array([[[u,v]]],np.float64),K,D if D.size else None).reshape(2)
    camera=np.r_[xy,1.]*z
    base=None
    if hand_eye is not None:
        T=compose_cam_to_base_transform(snapshot.T_base_end,hand_eye,cfg or {})
        if T.shape!=(4,4) or not np.isfinite(T).all():
            raise ValueError("手眼/末端变换无效")
        base=(T@np.r_[camera,1.])[:3]
    return camera,base


def project_button_hint(snapshot, point_base, hand_eye, cfg):
    """Project a previous measurement only to prioritize new OCR work."""
    camera_to_base=compose_cam_to_base_transform(snapshot.T_base_end,hand_eye,cfg)
    point=camera_to_base[:3,:3].T @ (np.asarray(point_base,float)-camera_to_base[:3,3])
    if not np.isfinite(point).all() or point[2]<=0.:
        return None
    zero=np.zeros(3)
    try:
        pixels,_=cv2.projectPoints(point.reshape(1,3),zero,zero,snapshot.K,snapshot.D)
    except cv2.error:
        return None
    u,v=pixels.reshape(2)
    h,w=snapshot.color.shape[:2]
    return (float(u),float(v)) if 0<=u<w and 0<=v<h else None


def measure_button_surface(snapshot, button, hand_eye, cfg, *, vertical_panel=False):
    """Fit the panel around the rim; use center depth for the button face.

    The vertical prior assumes base Z is upright. Apply it after the full
    rotation from the camera so wrist pitch is not mistaken for panel tilt.
    """
    camera, base = point_from_button(snapshot, button, hand_eye, cfg)
    if base is None:
        raise ValueError('估计面板法向需要手眼标定')
    ring = button.ring
    radius = float(ring.radius)
    if not math.isfinite(radius) or radius < 4.:
        raise ValueError('圆环过小或尺寸无效，无法可靠估计法向')
    depth = np.asarray(snapshot.depth_mm)
    u, v = ring.center
    height, width = depth.shape
    face = depth[max(0,v-3):v+4, max(0,u-3):u+4].ravel()
    face = face[np.isfinite(face) & (face >= 50.) & (face <= 1000.)]
    if len(face) < 9 or np.median(np.abs(face-1000.*camera[2])) > 4.:
        raise ValueError('按钮中央深度样本不足或不一致')
    angle = np.linspace(0., 2.*np.pi, 64, endpoint=False)
    offsets = np.linspace(1.18, 2.05, 5)[:, None, None] * radius * np.column_stack((np.cos(angle), np.sin(angle)))
    pixels = np.unique(np.rint(np.array([ring.x, ring.y])+offsets.reshape(-1, 2)).astype(int), axis=0)
    requested = len(pixels)
    pixels = pixels[((pixels >= 0) & (pixels < [width, height])).all(axis=1)]
    z = depth[pixels[:,1], pixels[:,0]] / 1000.
    valid = np.isfinite(z) & (z >= .05) & (z <= 1.) & (np.abs(z-camera[2]) <= .05)
    if valid.sum() < max(30, .45*requested):
        raise ValueError('圆环周边有效深度不足，无法可靠估计法向')
    pixels, z = pixels[valid], z[valid]
    distortion = np.asarray(snapshot.D, float)
    xy = cv2.undistortPoints(pixels.astype(float).reshape(-1,1,2), np.asarray(snapshot.K,float),
                            distortion if distortion.size else None).reshape(-1,2)
    points = np.column_stack((xy, np.ones(len(xy)))) * z[:, None]
    triples = points[np.random.default_rng(0).integers(len(points), size=(72,3))]
    axes = np.cross(triples[:,1]-triples[:,0], triples[:,2]-triples[:,0])
    lengths = np.linalg.norm(axes, axis=1)
    axes, origins = axes[lengths>1e-7]/lengths[lengths>1e-7,None], triples[lengths>1e-7,0]
    near = np.abs(np.einsum('ij,ij->i', camera-origins, axes)) <= .025
    axes, origins = axes[near], origins[near]
    if not len(axes) or not np.isfinite(points).all():
        raise ValueError('未找到按钮附近的有效面板平面')
    errors = np.abs(points @ axes.T - np.einsum('ij,ij->i', origins, axes))
    masks = errors <= .0035
    votes = masks.sum(axis=0)
    best = int(np.lexsort(((errors*masks).sum(axis=0), -votes))[0])
    inliers = masks[:, best]
    quadrants = (pixels[:,0] >= ring.x).astype(int) + 2*(pixels[:,1] >= ring.y)
    if votes[best] < max(30, .68*len(points)) or (np.bincount(quadrants[inliers], minlength=4)>=5).sum() < 3:
        raise ValueError('圆环周边不支持同一平面，拒绝猜测法向')
    samples = points[inliers]
    origin = samples.mean(axis=0)
    _, singular, basis = np.linalg.svd(samples-origin, full_matrices=False)
    normal = basis[-1]
    if normal @ origin > 0.:
        normal = -normal
    relief = float(normal @ (camera-origin))
    if singular[1]/np.sqrt(len(samples)) < .003 or not -.012 <= relief <= .025:
        raise ValueError('按钮深度与面板平面不一致')
    rotation = compose_cam_to_base_transform(snapshot.T_base_end, hand_eye, cfg)[:3,:3]
    outward = rotation @ normal
    length = np.linalg.norm(outward)
    if not math.isfinite(length) or length < 1e-8:
        raise ValueError('面板法向无法变换至 base 坐标')
    measured = outward/length
    outward = measured
    if vertical_panel:
        elevation = float(np.degrees(np.arcsin(np.clip(measured[2],-1.,1.))))
        if abs(elevation) > 20.:
            raise ValueError(f'实测法向偏离 base 水平面 {elevation:+.1f}°（超过20°）；请检查深度和底座是否水平')
        outward = measured.copy()
        outward[2] = 0.
        outward /= np.linalg.norm(outward)
    rms = float(np.sqrt(np.mean(((samples-origin)@normal)**2)))
    return ButtonSurface(snapshot,button,camera,base,outward,measured,rms,len(samples))


def pose_difference(first, second):
    relative = first[:3,:3].T @ second[:3,:3]
    angle = np.arccos(np.clip((np.trace(relative)-1.)/2.,-1.,1.))
    return np.linalg.norm(first[:3,3]-second[:3,3]), angle


def offset_target_base(point_base, offset):
    """Apply an elevator-only empirical correction AFTER the original transform.

    The correction is expressed in base XYZ, not camera/tool axes. A preview
    without a measured base pose still has no base target.
    """
    delta=np.asarray(offset,float)
    if delta.shape!=(3,) or not np.isfinite(delta).all():
        raise ValueError("elevator.target_offset_base_m需要三个有限米值")
    if point_base is None:return None
    point=np.asarray(point_base,float)
    if point.shape!=(3,) or not np.isfinite(point).all():
        raise ValueError("补偿前base目标无效")
    return point+delta


def check_cached_observation(snapshot, current, button, max_age_s=2.):
    """Reuse one RGB-D/pose bundle only while its scene and camera agree.

    Current pixels/depth are used for a cheap consistency check, not spliced
    into an older OCR observation. No inference or hardware access here.
    max_age_s=None retains a confirmed floor until new scene information;
    generation, timestamp ordering, image, depth and pose checks still apply.
    """
    age=current.captured_at-snapshot.captured_at
    if (snapshot.generation!=current.generation or not math.isfinite(age) or age<0
            or max_age_s is not None and not 0<=age<=max_age_s):
        raise ValueError("稳定图像已过期或运动批次已改变")
    old,new=np.asarray(snapshot.T_base_end),np.asarray(current.T_base_end)
    if old.shape!=(4,4) or new.shape!=(4,4) or not np.isfinite([old,new]).all():
        raise ValueError("缓存位姿无效")
    translation,angle=pose_difference(old,new)
    if translation>.002 or angle>np.radians(.5):
        raise ValueError("相机位姿已改变，等待新识别")
    if snapshot.color.shape!=current.color.shape or snapshot.depth_mm.shape!=current.depth_mm.shape:
        raise ValueError("缓存图像尺寸已改变")
    if not np.allclose(snapshot.K,current.K) or not np.array_equal(snapshot.D,current.D):
        raise ValueError("缓存内参已改变")
    u,v=button.ring.center
    radius=max(8,int(button.ring.radius*1.3))
    h,w=current.color.shape[:2]
    x1,y1,x2,y2=max(0,u-radius),max(0,v-radius),min(w,u+radius+1),min(h,v+radius+1)
    if x2<=x1 or y2<=y1:
        raise ValueError("缓存圆环不在画面内")
    patches=[cv2.cvtColor(s.color[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY) for s in (snapshot,current)]
    difference=cv2.absdiff(*patches)
    if np.mean(difference)>8. or np.percentile(difference,95)>25.:
        raise ValueError("圆环图像已改变，等待新识别")
    depths=[get_depth_mm(s.depth_mm,u,v,roi_size=7) for s in (snapshot,current)]
    if not np.isfinite(depths).all() or min(depths)<=0 or abs(depths[1]-depths[0])>3.:
        raise ValueError("圆环深度已改变或无效，等待新识别")


def level_pose(current, tip_offset=(0.,0.,0.)):
    """Base end_link roll=pitch=0; preserve current yaw and tool-tip position."""
    current=np.asarray(current,float);offset=np.asarray(tip_offset,float)
    if current.shape!=(4,4) or offset.shape!=(3,) or not all(np.isfinite(x).all() for x in (current,offset)):
        raise ValueError("水平姿态输入无效")
    tip=current[:3,3]+current[:3,:3]@offset
    yaw=math.atan2(current[1,0],current[0,0])
    T=pose6d_to_mat4(0.,0.,0.,0.,0.,yaw)
    T[:3,3]=tip-T[:3,:3]@offset
    return T


def panel_facing_pose(outward):
    """Keep the tool upright and point end_link +X into the panel in base XY."""
    normal=np.asarray(outward,float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal[:2])<1e-8:
        raise ValueError('面板法向缺少有效的水平分量')
    yaw=math.atan2(-normal[1],-normal[0])
    return pose6d_to_mat4(0.,0.,0.,0.,0.,yaw)


def horizontal_approach_direction(current):
    """Keep the established horizontal approach heading while freeing wrist tilt."""
    T=np.asarray(current,float)
    if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('接近方向需要有效末端位姿')
    direction=np.r_[T[:2,0],0.]
    length=np.linalg.norm(direction)
    return direction/length if length>1e-8 else np.array([1.,0.,0.])


def button_poses(point_base, leveled_pose, tip_offset, standoff=.05, press_travel=0., *, forward_base=None):
    """Tip targets along tool +X or an explicit base direction, then same retreat.

    Position IK callers treat the matrix rotation as a placeholder and constrain
    only translation + rotation @ tip_offset, never this placeholder rotation.
    """
    point=np.asarray(point_base,float);T=np.asarray(leveled_pose,float);offset=np.asarray(tip_offset,float)
    if point.shape!=(3,) or T.shape!=(4,4) or offset.shape!=(3,) or not all(np.isfinite(x).all() for x in (point,T,offset)):
        raise ValueError("按钮目标无效")
    if not .03<=standoff<=.25 or not 0<=press_travel<=.005:
        raise ValueError("前置距离需3～25cm；实体按钮实测压入量需0～5mm")
    forward=T[:3,0] if forward_base is None else np.asarray(forward_base,float)
    if forward.shape!=(3,) or not np.isfinite(forward).all() or np.linalg.norm(forward)<1e-8:
        raise ValueError('接近方向需要非零的三维有限向量')
    forward=forward/np.linalg.norm(forward)
    def at(distance):
        return pose_for_tip(T, point+forward*distance, offset)
    return dict(approach=at(-standoff),surface=at(0.),press=at(press_travel),retreat=at(-standoff))


def pose_for_tip(pose, tip_base, tip_offset):
    result = np.asarray(pose,float).copy()
    result[:3,3] = np.asarray(tip_base,float)-result[:3,:3]@np.asarray(tip_offset,float)
    return result


def tip_position(pose, tip_offset):
    return pose[:3,3]+pose[:3,:3]@np.asarray(tip_offset,float)
