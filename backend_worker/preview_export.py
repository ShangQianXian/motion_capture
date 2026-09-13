"""Capture actual detector observations alongside the final 3D result."""

from __future__ import annotations

import hashlib
import json
import math
import os

from ._core import preview, skeleton


def decoded_row(decoded, frame_start):
    return {"source_index": decoded.source_index, "time": decoded.timestamp,
            "sample_frame": frame_start + decoded.index, "result_frame": None,
            "body2d": [], "hands2d": [], "status": "missing"}


def mp_points(landmarks):
    points = []
    for lm in landmarks:
        score = getattr(lm, "visibility", None)
        if score is None:
            score = getattr(lm, "presence", None)
        values = (float(lm.x), float(lm.y), float(1.0 if score is None else score))
        points.append([values[0], values[1], max(0.0, min(1.0, values[2]))]
                      if all(math.isfinite(x) for x in values) else [0.0, 0.0, 0.0])
    return points


def coco_points(points, scores, width, height):
    output = []
    for point, score in zip(points, scores):
        x, y, confidence = float(point[0]) / width, float(point[1]) / height, float(score)
        output.append([x, y, max(0.0, min(1.0, confidence))]
                      if all(math.isfinite(v) for v in (x, y, confidence)) else [0.0, 0.0, 0.0])
    return output


def source_meta(source):
    return {"fps": source.native_fps, "effective_fps": source.fps,
            "width": source.width, "height": source.height,
            "total_frames": source.native_total, "type": source.media_type}


def write(job, result, result_path, rows, meta, profile, raw_frames=None, diagnostics=None):
    if not rows:
        return
    by_frame = {frame["frame"]: frame for frame in result["frames"]}
    estimated = ['toe.L', 'toe.R'] if profile in ('quality', 'quality_plus', 'quality_feet') else []
    for row in rows:
        frame = by_frame.get(row["sample_frame"])
        if frame:
            row["result_frame"] = frame["frame"]
            scores = frame.get("confidence", {})
            low = [name for name, value in scores.items() if isinstance(value, (int, float))
                   and value < skeleton.CONFIDENCE_LOW and name not in estimated]
            row["low_joints"] = low
            row["estimated_joints"] = estimated
            row['unreliable_joints'] = [name for name in row.get('unreliable_joints', []) if name not in estimated]
            # postprocess interpolates per-joint tracks with low confidence.
            interpolated = [name for name in frame.get("interpolated_joints", row.get('interpolated_joints', [])) if name not in estimated]
            row["interpolated_joints"] = interpolated
            row["status"] = "interpolated" if interpolated else ("low_confidence" if low else "detected")
        else:
            row["status"] = "missing"
    with open(result_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    source = preview.fingerprint(job["input"]["path"])
    source.update(meta)
    payload = {"version": preview.VERSION, "job_id": job["job_id"], "source": source,
               "capture_settings": job.get("review_settings", {}),
               "options": dict(job.get("options") or {}), "input": dict(job["input"]),
               "profile": profile, "result_sha256": digest,
               "topology": "mediapipe33" if profile in ("preview", "fallback_cpu") else "coco17",
               "edges": preview.MP_EDGES if profile in ("preview", "fallback_cpu") else preview.COCO_EDGES,
               "frames": rows}
    if profile == 'quality_feet':
        payload['topology'] = 'coco_wholebody133'
        payload['edges'] = list(preview.COCO_EDGES) + [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
    if diagnostics:
        payload['diagnostics'] = diagnostics
        payload['processing_version'] = '0.3'
        payload['coordinate_space'] = diagnostics.get('coordinate_space', 'root_relative')
        payload['motion_type'] = diagnostics.get('motion_type', 'general')
    if raw_frames:
        from . import export_result
        raw = export_result.build_result(raw_frames, result['fps'], source_path=job['input']['path'],
                                         source_type=job['input']['type'], profile=profile)
        path = export_result.write_result(raw, os.path.dirname(result_path), 'mocap_raw.json')
        with open(path, 'rb') as handle:
            payload['stages'] = {'raw': {'file': 'mocap_raw.json', 'sha256': hashlib.sha256(handle.read()).hexdigest()}}
        raw_numbers = {frame['frame'] for frame in raw_frames}
        for row in rows:
            row['raw_result_frame'] = row['sample_frame'] if row['sample_frame'] in raw_numbers else None
            row['joint_sources'] = {'body3d': 'mediapipe_world' if profile in ('preview', 'fallback_cpu') else 'motionbert_lifted',
                                    'processed_pelvis': 'kinematic_fit',
                                    'feet3d': 'estimated' if estimated else 'mediapipe_world',
                                    'feet2d': 'detected' if row.get('feet2d') else 'unavailable'}
    preview.validate_manifest(payload)
    target = os.path.join(os.path.dirname(result_path), preview.MANIFEST_FILENAME)
    with open(target + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    os.replace(target + ".tmp", target)
