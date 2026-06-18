#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# detector.py – pre-filter person detection (HOG and YOLO)


def detect(detector_type, image_path):
    if detector_type == "hog":
        return _hog(image_path)
    elif detector_type == "yolo":
        return _yolo(image_path)
    return {"count": 0, "detections": []}


def _hog(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found: " + image_path)
    s = 800 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img.copy()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
    return {
        "count": len(boxes),
        "detections": [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "confidence": float(weights[i])}
            for i, (x, y, w, h) in enumerate(boxes)
        ],
    }


# YOLO runs in a child process so PyTorch/OpenSL ES resources are released
# before the caller plays audio or runs other commands.
_YOLO_SCRIPT = r"""
import sys, json, os, warnings
warnings.filterwarnings('ignore')
os.environ['YOLO_VERBOSE'] = '0'

import cv2

# silence ultralytics loader output by temporarily redirecting stdout
import io as _io
_orig_stdout = sys.stdout
sys.stdout = _io.StringIO()
try:
    from ultralytics import YOLO
    _model = YOLO('yolov8n.pt')
finally:
    sys.stdout = _orig_stdout

_path = sys.argv[1]
_img = cv2.imread(_path)
if _img is None:
    print(json.dumps({'error': 'Image not found: ' + _path}), flush=True)
    sys.exit(1)

_results = _model(_img, verbose=False)
_persons = [
    float(b.conf[0]) for b in _results[0].boxes
    if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5
]
print(json.dumps({'count': len(_persons),
                  'detections': [{'confidence': k} for k in _persons]}), flush=True)
"""


def _yolo(image_path):
    import sys
    import json
    import subprocess
    import os

    env = dict(os.environ)
    env["YOLO_VERBOSE"] = "0"

    proc = subprocess.run(
        [sys.executable, "-c", _YOLO_SCRIPT, image_path],
        capture_output=True, text=True, env=env
    )

    # pick the last line that looks like JSON (ignore any stray prints)
    json_lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if not json_lines:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            "YOLO subprocess failed (rc={}){}".format(
                proc.returncode, ": " + detail if detail else ""
            )
        )

    data = json.loads(json_lines[-1])
    if "error" in data:
        raise ValueError(data["error"])
    return data
