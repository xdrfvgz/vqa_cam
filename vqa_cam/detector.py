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


def _yolo(image_path):
    import sys
    import json
    import subprocess
    # Run YOLO in an isolated child process so its PyTorch/OpenSL ES resources
    # are fully released before the caller runs play-audio or other audio commands.
    # stdout carries only our JSON line; all other output goes to stderr.
    _SCRIPT = "\n".join([
        "import sys, json, os, warnings",
        "warnings.filterwarnings('ignore')",
        "os.environ['YOLO_VERBOSE'] = '0'",
        "os.environ['ULTRALYTICS_LOGGING'] = '0'",
        "import contextlib, io",
        "import cv2",
        "_real_stdout = sys.stdout",
        "sys.stdout = io.StringIO()",   # silence any loader prints
        "from ultralytics import YOLO",
        "model = YOLO('yolov8n.pt')",
        "sys.stdout = _real_stdout",
        "path = sys.argv[1]",
        "img = cv2.imread(path)",
        "if img is None:",
        "    print(json.dumps({'error': 'Image not found: ' + path}))",
        "    sys.exit(1)",
        "r = model(img, verbose=False)",
        "p = [float(b.conf[0]) for b in r[0].boxes",
        "     if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5]",
        "print(json.dumps({'count': len(p), 'detections': [{'confidence': k} for k in p]}))",
    ])
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, image_path],
        capture_output=True, text=True
    )
    # find the last line that looks like JSON (ignores any stray prints)
    json_lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    if not json_lines:
        err = proc.stderr.strip() or proc.stdout.strip() or "YOLO subprocess produced no output"
        raise RuntimeError(err)
    data = json.loads(json_lines[-1])
    if "error" in data:
        raise ValueError(data["error"])
    return data
